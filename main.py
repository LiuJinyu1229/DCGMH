import argparse
import utils
from engine import *
import random

def load_config():
    parser = argparse.ArgumentParser(description='PSLDH_PyTorch')
    parser.add_argument('--dataset', default = 'flickr', choices=['coco', 'nuswide', 'flickr', 'iaprtc'], help='Dataset name')
    parser.add_argument('--bit', default=16, type=int, help='Binary hash code length')
    parser.add_argument('--num_class', default=24, type=int, help='The number of classes')
    parser.add_argument('--epochs', default=150, type=int, help='Number of iterations')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of iterations')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of loading data threads')
    parser.add_argument('--batch_size', default=48, type=int, help='the batch size for training')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--evaluate', action='store_true', help='Evaluation mode')
    parser.add_argument('--gpu', default=1, type=int, help='define gpu id')
    parser.add_argument('--noise', default=True, type=bool, help='Whwarmether to add noise to label')
    parser.add_argument('--noise_type', default='symmetric', type=str, help='pairflip or symmetric')
    parser.add_argument('--noise_level', default=0.4, type=float, help='Noise Level')
    parser.add_argument('--checkpoint', default='./checkpoint/model.pkl', type=str, help='the saved parameters of model')
    parser.add_argument('--img_hidden_dim', type=list, default=[4096, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[4096, 128], help='Construct textMLP')
    parser.add_argument('--alpha', default=1, type=float, help='parameter for quantify_loss')
    parser.add_argument('--beta', default=0.15, type=float, help='parameter for unsupervised_consistency_loss')
    parser.add_argument('--gamma', default=5, type=float, help='parameter for center_loss')
    parser.add_argument('--eta', default=1, type=float, help='parameter for reconstruct_loss')
    parser.add_argument('--step_size', default=100, type=int, help='step size of lr_scheduler')
    parser.add_argument('--margin', default=0.7, type=float, help='margin for contrastive loss')
    parser.add_argument('--seed', default=2024, type=int, help='random seed')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = load_config()
    logger = utils.logger(args)
    utils.log_params(logger, vars(args))
    # seed = random.randint(0, 10000)
    seed = args.seed
    random_seed = seed
    logger.info(f'Random Seed: {random_seed}')
    utils.seed_setting(seed=random_seed) 
    engine = Engine(args=args, logger=logger)

    if args.train:
        # engine.warmup()
        engine.train()
        del engine
    elif args.evaluate:
        engine.evaluate()
    else:
        raise ValueError('Error configuration, please check your config, using "train" or "evaluate".')
