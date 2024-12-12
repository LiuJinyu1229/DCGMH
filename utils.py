import logging
import os
import random
import time

import numpy as np
import torch

def logger(option):
    '''
    '\033[0;34m%s\033[0m': blue
    :return:
    '''
    logger = logging.getLogger('PAGN')
    logger.setLevel(logging.DEBUG)

    if not os.path.exists('log/' + option.dataset + '/'):
        os.mkdir('log/' + option.dataset + '/')

    file_name = time.strftime("[%m-%d] [%H:%M:%S]", time.localtime()) + str(option.bit) + "_lr" + str(option.lr) + '_' + str(option.alpha) + '_' + str(option.beta) + '_' + str(option.gamma)  + '_' + str(option.eta)

    txt_log = logging.FileHandler('log/' + option.dataset + '/' + file_name +'.log', mode='a')
    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    # console + color
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('\033[0;32m%s\033[0m' % '[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger

def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False # False make training process too slow!
    torch.backends.cudnn.deterministic = True

def log_params(logger, config: dict):
    logger.info('--- Configs List---')
    for k in config.keys():
        logger.info('--- {:<18}:{}'.format(k, config[k]))

def label_similarity(label_1, label_2):
    aff = torch.matmul(label_1, label_2.T)
    affinity_matrix = aff.float()
    affinity_matrix = 1 / (1 + torch.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    return affinity_matrix

def calculate_map(te_B, re_B, te_L, re_L, batch_size=10):
    """
    :param te_B: {-1,+1}^{mxq} test hash code (torch tensor)
    :param re_B: {-1,+1}^{nxq} retrieval hash code (torch tensor)
    :param te_L: {0,1}^{mxl} test label (torch tensor)
    :param re_L: {0,1}^{nxl} retrieval label (torch tensor)
    :param batch_size: size of the batch to process (int)
    :return: mean average precision (float)
    """
    te_B = te_B.cuda()
    re_B = re_B.cuda()
    te_L = te_L.cuda()
    re_L = re_L.cuda()
    
    num_test = te_L.shape[0]
    num_retrieval = re_L.shape[0]
    map_score = 0.0
    num_batches = (num_test + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_test)
        
        te_B_batch = te_B[start_idx:end_idx]
        te_L_batch = te_L[start_idx:end_idx]
        
        hamm = calculate_hamming(te_B_batch, re_B)
        gnd = torch.matmul(te_L_batch, re_L.transpose(0, 1)) > 0
        gnd = gnd.float().cuda()
        
        tsum = torch.sum(gnd, dim=1)
        valid_mask = tsum > 0
        
        if torch.any(valid_mask):
            sorted_indices = torch.argsort(hamm, dim=1)
            sorted_gnd = torch.gather(gnd, 1, sorted_indices)
            
            cumulative_gnd = torch.cumsum(sorted_gnd, dim=1)
            valid_cumulative_gnd = cumulative_gnd[valid_mask]
            
            row_indices = torch.arange(1, sorted_gnd.size(1) + 1, device=te_B.device).float().view(1, -1)
            valid_row_indices = row_indices.repeat(valid_cumulative_gnd.size(0), 1)
            
            precisions = valid_cumulative_gnd / valid_row_indices
            batch_map_score = torch.sum(precisions * sorted_gnd[valid_mask], dim=1) / tsum[valid_mask]
            
            map_score += torch.sum(batch_map_score).item()
        
        del te_B_batch, te_L_batch, hamm, gnd, tsum, valid_mask, sorted_indices, sorted_gnd, cumulative_gnd, valid_cumulative_gnd, row_indices, valid_row_indices, precisions
        torch.cuda.empty_cache()
    
    map_score /= num_test
    
    return map_score

def calculate_hamming(B1, B2):
    """
    :param B1: vector [q] (torch tensor)
    :param B2: matrix [n x q] (torch tensor)
    :return: hamming distance [n] (torch tensor)
    """
    length = B2.shape[1]  # max inner product value
    distH = 0.5 * (length - torch.matmul(B1, B2.transpose(0, 1)).cuda())
    return distH

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def pr_curve(qB, rB, queryL, retrievalL):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    dim = np.shape(rB)
    bit = dim[1]
    all_ = dim[0]
    precision = np.zeros(bit + 1)
    recall = np.zeros(bit + 1)
    num_query = queryL.shape[0]
    num_database = retrievalL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        all_sum = np.sum(gnd).astype(np.float32)
        # print(all_sum)
        if all_sum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        # print(hamm.shape)
        ind = np.argsort(hamm)
        # print(ind.shape)
        gnd = gnd[ind]
        hamm = hamm[ind]
        hamm = hamm.tolist()
        # print(len(hamm), num_database - 1)
        max_ = hamm[num_database - 1]
        max_ = int(max_)
        t = 0
        for i in range(1, max_):
            if i in hamm:
                idd = hamm.index(i)
                if idd != 0:
                    sum1 = np.sum(gnd[:idd])
                    precision[t] += sum1 / idd
                    recall[t] += sum1 / all_sum
                else:
                    precision[t] += 0
                    recall[t] += 0
                t += 1
        # precision[t] += all_sum / num_database
        # recall[t] += 1
        for i in range(t,  bit + 1):
            precision[i] += all_sum / num_database
            recall[i] += 1
    true_recall = recall / num_query
    precision = precision / num_query
    print(true_recall)
    print(precision)
    return true_recall, precision

def precision_topn(qB, rB, queryL, retrievalL, topk=1000):
    n = topk // 100
    precision = np.zeros(n)
    num_query = queryL.shape[0]
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        for i in range(1, n + 1):
            a = gnd[:i * 100]
            precision[i - 1] += float(a.sum()) / (i * 100.)
    a_precision = precision / num_query
    return a_precision
