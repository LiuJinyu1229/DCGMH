import pickle
import torch.nn.functional as F
import torch.nn as nn
import scipy.io as sio
import os
from data_loader import *
import utils
from model.fusion import *
from model.label import *
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from collections import Counter
from data_loader import image_augmentations, text_augmentations

class Engine(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.dataset = args.dataset
        self.noise = args.noise
        self.noise_type = args.noise_type
        self.noise_level = args.noise_level
        self.lr = args.lr
        self.bit = args.bit
        self.num_class = args.num_class
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.warmup_epochs = args.warmup_epochs
        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.step_size = args.step_size
        self.margin = args.margin
        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        self.train_dataset = DatasetProcess(self.dataset, 'train', self.noise, self.noise_type, self.noise_level)
        self.test_dataset = DatasetProcess(self.dataset, 'test')
        self.retrieval_dataset = DatasetProcess(self.dataset, 'retrieval')

        self.num_train = len(self.train_dataset)
        self.num_test = len(self.test_dataset)
        self.num_retrieval = len(self.retrieval_dataset)
        self.train_labels = self.train_dataset.nlabels
        self.flags = self.train_dataset.flags
        self.labels = self.train_dataset.labels
        self.test_labels = self.test_dataset.labels
        self.retrieval_labels = self.retrieval_dataset.labels
        self.image_dim = self.train_dataset.imgs.shape[1]
        self.text_dim = self.train_dataset.txts.shape[1]
        self.label_dim = self.num_class

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.eta = args.eta

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.retrieval_loader = DataLoader(self.retrieval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.img_hidden_dim.insert(0, self.image_dim)
        self.img_hidden_dim[-1] = self.img_hidden_dim[-1] * self.bit
        self.txt_hidden_dim.insert(0, self.text_dim)
        self.txt_hidden_dim[-1] = self.txt_hidden_dim[-1] * self.bit

        self.hash_net = Hash_Net(self.img_hidden_dim, self.txt_hidden_dim, self.bit, self.label_dim)
        self.optimizer_hash = optim.SGD(self.hash_net.parameters(), lr=self.lr, momentum=0.9, weight_decay=10 ** -5)
        self.scheduler_hash = torch.optim.lr_scheduler.StepLR(self.optimizer_hash, step_size=self.step_size, gamma=0.1, last_epoch=-1)

        self.label_code = None
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.testCode = None
        self.retrievalCode = None
        self.map = 0
    
    def code_center_loss(self, hash_code, center, label, eps = 1e-5):
        code_length = hash_code.shape[1]
        logit_ii = hash_code.mm(center.t()) / code_length
        our_logit_ii = torch.exp(logit_ii) * label
        mu_logit_ii = (torch.exp(logit_ii) * (1 - label)).sum(1).view(-1, 1).expand(logit_ii.shape[0], logit_ii.shape[1]) + our_logit_ii
        lossi = -((torch.log((our_logit_ii) / (mu_logit_ii + eps) + eps) * label).sum(1) / label.sum(1))
        loss = lossi.mean()
        return loss, lossi

    def reconstruct_loss(self, hash_code, center, label):
        code_center = hash_code.mm(center.t())
        loss = self.bce_loss(code_center, label)
        return loss
    
    def contrastive_loss(self, hash1, hash2, margin=0.7): # 0.3
        batch_size = hash1.size(0)

        hash1 = F.normalize(hash1, p=2, dim=1)
        hash2 = F.normalize(hash2, p=2, dim=1)

        similarity_matrix = F.cosine_similarity(hash1.unsqueeze(1), hash2.unsqueeze(0), dim=-1)
        mask = torch.eye(batch_size, dtype=torch.bool).cuda()

        positive_loss = 1 - torch.diagonal(similarity_matrix)
        negative_loss = F.relu(similarity_matrix - margin)
        negative_loss = negative_loss * (~mask)

        loss = positive_loss.mean() + negative_loss.mean()
        return loss
    
    def center_loss(self, centroids):
        centroids_dist = torch.cdist(centroids, centroids, p=2) / centroids.shape[1]
        triu_dist = torch.triu(centroids_dist, diagonal=1)
        mean_dist = -torch.mean(triu_dist)
        triu_dist_inf = torch.where(triu_dist == 0, float('inf'), triu_dist)
        min_dist = -torch.min(triu_dist_inf)
        reg_term = mean_dist + min_dist
        return reg_term
    
    def GEN_S_GPU(self, label_1, label_2):
        aff = torch.matmul(label_1, label_2.T)
        affinity_matrix = aff.float()
        affinity_matrix = 1 / (1 + torch.exp(-affinity_matrix))
        affinity_matrix = 2 * affinity_matrix - 1
        return affinity_matrix
    
    def identify(self, sim_code_center, nlabel, label, noise_level): # ours
    # def identify(self, hash_code, hash_code_aug_v, center, nlabel, label, noise_level): # DIOR 
        judge = torch.any(nlabel != label, dim=1).float().cuda()
        indice = torch.nonzero(judge == 1).squeeze().cuda()

        # DIOR 
        # _, lossi = self.code_center_loss(hash_code, center, nlabel)
        # _, lossi_aug_v = self.code_center_loss(hash_code_aug_v, center, nlabel)
        # difference_batch = torch.abs(lossi - lossi_aug_v)
        # _, indice_consistency = torch.sort(difference_batch, descending=True)

        # ours
        consistency = torch.sum(nlabel * sim_code_center, dim=-1) / nlabel.sum(1)
        _, indice_consistency = torch.sort(consistency, descending=False)
        select_noisy_sim = indice_consistency[:int(len(indice_consistency) * noise_level)]
        select_cor_sim = indice_consistency[int(len(indice_consistency) * noise_level):]

        return select_noisy_sim, select_cor_sim
    
    def reconstruct(self, noisy_code_center, code_center, noisy_indice, noisy_or_not, label, k=2):
        noisy_code_sim = noisy_code_center.mm(code_center.t())
        top_values, top_indices = torch.topk(noisy_code_sim, k=k, dim=1)
        reconstruct_label = torch.zeros(top_indices.shape[0], self.train_labels.shape[1]).cuda()
        for i in range(top_indices.shape[0]):
            valid_indices = top_indices[i][noisy_or_not[top_indices[i]].squeeze().bool()]
            if len(valid_indices) > 0:
                select_labels = self.train_labels[valid_indices]
                select_flags = self.flags[valid_indices]
                unique_flags, counts = torch.unique(select_flags, return_counts=True)
                max_count = counts.max()
                if max_count == k:
                    most_common_flag = unique_flags[counts.argmax()]
                    reconstruct_label[i] = select_labels[select_flags == most_common_flag][0]
        
        # divide the reconstruct samples and unsupervised samples
        zero_mask = (reconstruct_label.sum(dim=1) == 0)
        zero_indices = torch.nonzero(zero_mask, as_tuple=False).squeeze()
        unsupervised_indice = noisy_indice[zero_indices]
        non_zero_mask = (reconstruct_label.sum(dim=1) > 0)
        non_zero_indices = torch.nonzero(non_zero_mask, as_tuple=False).squeeze()
        reconstruct_indice = noisy_indice[non_zero_indices]
        final_reconstruct_label = reconstruct_label[non_zero_mask]

        return final_reconstruct_label, reconstruct_indice, unsupervised_indice

    def train(self):
        self.logger.info("start to training...")
        self.hash_net.train().cuda()
        self.train_labels = torch.tensor(self.train_labels).cuda()
        self.flags = torch.tensor(self.flags).cuda()
        code_center = torch.randn((self.train_labels.shape[0], self.num_class)).cuda()
        noisy_or_not = torch.ones((self.train_labels.shape[0], 1)).cuda()
        for epoch in range(self.epochs):
            self.scheduler_hash.step()
            epoch_loss = 0.
            epoch_radio = []
            reconstruct_num = 0
            for batch_idx, (img, txt, label, nlabel, index) in tqdm(enumerate(self.train_loader)):
                img = img.cuda()
                txt = txt.cuda()
                label = label.cuda()
                nlabel = nlabel.cuda()
                index = index.cuda()
                img_aug_v = image_augmentations(img).cuda()
                txt_aug_v = text_augmentations(txt).cuda()
                hash_code = self.hash_net(img, txt)
                hash_code_aug_v = self.hash_net(img_aug_v, txt_aug_v)
                hash_code_norm = F.normalize(hash_code, p=2, dim=1)
                center = self.hash_net.centroids.to(dtype=torch.float32).cuda()
                center_norm = F.normalize(center, p=2, dim=1)

                sim_code_center = hash_code_norm.mm(center_norm.t()).cuda()

                if epoch < self.warmup_epochs:
                    H = torch.sign(hash_code)
                    quantify_loss = self.mse_loss(hash_code, H)
                    center_loss = self.center_loss(center)
                    code_center_loss, _ = self.code_center_loss(hash_code, center, nlabel)
                    loss = code_center_loss + self.alpha * quantify_loss + self.gamma * center_loss

                else:
                    H = torch.sign(hash_code)
                    quantify_loss = self.mse_loss(hash_code, H)
                    center_loss = self.center_loss(center)

                    # filter noisy samples
                    noisy_indice, correct_indice = self.identify(sim_code_center, nlabel, label, self.noise_level)
                    # noisy_indice, correct_indice = self.identify(hash_code, hash_code_aug_v, center, nlabel, label, self.noise_level)
                    code_center[index[correct_indice]] = sim_code_center[correct_indice]
                    code_center[index[noisy_indice]] = torch.zeros((noisy_indice.shape[0], self.num_class)).cuda()
                    noisy_or_not[index[noisy_indice]] = 0

                    # reconstrct high-confidence samples
                    noisy_code_center = sim_code_center[noisy_indice]
                    reconstruct_label, reconstruct_indice, unsupervised_indice = self.reconstruct(noisy_code_center, code_center, noisy_indice, noisy_or_not, label)

                    # correct label
                    code_correct = hash_code[correct_indice]
                    label_correct = nlabel[correct_indice]
                    correct_loss, _ = self.code_center_loss(code_correct, center, label_correct)

                    # reconstruct label
                    if reconstruct_indice.numel() > 0:
                        reconstruct_num += reconstruct_indice.numel()
                        code_reconstruct = hash_code[reconstruct_indice].unsqueeze(0) if reconstruct_indice.numel() == 1 else hash_code[reconstruct_indice]
                        code_reconstruct_norm = F.normalize(code_reconstruct)
                        label_reconstruct = reconstruct_label
                        reconstruct_loss = self.mse_loss(code_reconstruct_norm.mm(code_reconstruct_norm.t()), utils.label_similarity(label_reconstruct, label_reconstruct))
                    else:
                        reconstruct_loss = torch.tensor(0.0, requires_grad=True)
                    # reconstruct_loss = torch.tensor(0.0, requires_grad=True)
                    # unsupervised_indice = noisy_indice
                    
                    # unsupervised label
                    if unsupervised_indice.numel() > -1:
                        try:
                            code_unsupervised = hash_code[unsupervised_indice].unsqueeze(0) if unsupervised_indice.numel() == 1 else hash_code[unsupervised_indice]
                            img_aug = image_augmentations(img[unsupervised_indice].unsqueeze(0) if unsupervised_indice.numel() == 1 else img[unsupervised_indice])
                            txt_aug = text_augmentations(txt[unsupervised_indice].unsqueeze(0) if unsupervised_indice.numel() == 1 else txt[unsupervised_indice])
                            hash_code_aug = self.hash_net(img_aug, txt_aug)
                            contrastive_loss = self.contrastive_loss(hash_code_aug, code_unsupervised, self.margin)
                        except:
                            import ipdb
                            ipdb.set_trace()
                    else:
                        contrastive_loss = torch.tensor(0.0, requires_grad=True)
                    # contrastive_loss = torch.tensor(0.0, requires_grad=True)

                    loss = correct_loss + self.alpha * quantify_loss + self.beta * contrastive_loss + self.gamma * center_loss + self.eta * reconstruct_loss

                self.optimizer_hash.zero_grad()
                loss.backward()
                self.optimizer_hash.step()

                epoch_loss = epoch_loss + loss.item()
                if batch_idx + 1 == len(self.train_loader):
                    if epoch < self.warmup_epochs:
                        self.logger.info('[%4d/%4d] Loss: %.4f quantify_loss: %.4f code_center_loss: %.4f center_loss: %.4f' % \
                            (epoch + 1, self.epochs, loss.item(), quantify_loss.item(), code_center_loss.item(), center_loss.item()))
                    else:
                        self.logger.info('[%4d/%4d] Loss: %.4f quantify_loss: %.4f correct_loss: %.4f reconstruct_loss: %.4f contrastive_loss: %.4f center_loss: %.4f' \
                            % (epoch + 1, self.epochs, loss.item(), quantify_loss.item(), correct_loss.item(), reconstruct_loss, \
                            contrastive_loss, center_loss.item()))
            if (epoch + 1) % 5 == 0:
                self.evaluate()
                
        self.evaluate()

    def evaluate(self):
        self.logger.info("start to evaluating...")
        self.hash_net.eval()
        
        self.logger.info('Test size: %d' % (self.test_dataset.labels.shape[0]))  
        testP = []
        start_time = time.time() * 1000
        for i, (img, txt, label, index) in tqdm(enumerate(self.test_loader)):
            img = img.cuda()
            txt = txt.cuda()
            H = self.hash_net(img, txt)
            testP.append(H.data.cpu())
        testH = torch.cat(testP, dim=0).cuda()
        self.testCode = torch.sign(testH)
        end_time = time.time() * 1000
        test_time = end_time - start_time

        self.logger.info('Retrieval size: %d' % (self.retrieval_dataset.labels.shape[0])) 
        retrievalP = []
        start_time = time.time() * 1000
        for i, (img, txt, label, index) in tqdm(enumerate(self.retrieval_loader)):
            img = img.cuda()
            txt = txt.cuda()
            H = self.hash_net(img, txt)
            retrievalP.append(H.data.cpu())
        retrievalH = torch.cat(retrievalP, dim=0).cuda()
        self.retrievalCode = torch.sign(retrievalH)
        end_time = time.time() * 1000
        retrieval_time = end_time - start_time

        self.logger.info('[Test time] %.4f, [Retrieval time] %.4f' % (test_time / 1000, retrieval_time / 1000))

        self.map = utils.calculate_map(self.testCode, self.retrievalCode, self.test_labels, self.retrieval_labels)
        self.logger.info('MAP: %.4f' % self.map)
