import torch
import torch.backends.cudnn as cudnn

import os
import ast
import random
import numpy as np
import global_vars as gv
import argparse
from tqdm import tqdm
import copy

from utils.data_split import get_dataset
from utils.utils import Logger, per_class_acc
from utils.setup_NSL import NSL_KDD, NSL_data
from model import generate_model
from models import mlp

from train_local_model import LocalUpdate
from utils.federated_utils import average_weights, test_inference


def main(args):
    # if not os.path.exists(args.checkpoint_folder):
    #     os.makedirs(args.checkpoint_folder)
    # if not os.path.exists(args.result_folder):
    #     os.mkdir(args.result_folder)
    # if not os.path.exists(args.log_folder):
    #     os.makedirs(args.log_folder)
    # if not os.path.exists(args.normvec_folder):
    #     os.makedirs(args.normvec_folder)
    # if not os.path.exists(args.score_folder):
    #     os.makedirs(args.score_folder)

    # load dataset and user group
    if args.dataset == 'nsl':
        attack_type = {'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}

        if args.data_partition_type is "normalOverAll":
            all_data = NSL_KDD(data_type=None)
            normal_data = NSL_KDD(data_type='normal')
            anormal_data = NSL_KDD(data_type='anomaly')
        else:
            attack = [(args.data_partition_type, attack_type[args.data_partition_type])]
            all_data = NSL_KDD(attack, data_type=None)
            normal_data = NSL_KDD(attack, data_type='normal')
            anormal_data = NSL_KDD(attack, data_type='anomaly')

    train_normal, train_anormal, valid_data, test_data, user_groups_normal, user_groups_anormal = get_dataset(
        args, all_data, normal_data, anormal_data)

    _, test_labels = test_data[:]
    _, valid_labels = valid_data[:]
    valid_normal = NSL_data(normal_data.validation_data, normal_data.validation_labels)

    if not os.path.exists(args.checkpoint_folder):
        os.mkdir(args.checkpoint_folder)
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)

    # setup logger
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                          args.log_resume)

    if args.mode is 'train':
        for c in range(args.num_users):
            print(f'\n-------------------- Local Training of Client {c} Starts!! ----------------------')
            # BUILD MODEL
            model = generate_model(args, input_size=all_data.train_data.shape[1])
            model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
            if args.use_cuda:
                model_head.cuda()

            # initial
            best_acc = 0
            memory_bank = []
            checkpoint_path = os.path.join(args.checkpoint_folder,
                                           f'client{c}_best_model_{args.model_type}.pth')
            head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                f'client{c}_best_model_{args.model_type}_head.pth')

            for epoch in tqdm(range(args.epochs)):

                print(f'\n | Training Round : {epoch + 1} |\n')
                local_model = LocalUpdate(args, idxs_normal=user_groups_normal[c],
                                          idxs_anormal=user_groups_anormal[c],
                                          dataset_normal=train_normal,
                                          dataset_anormal=train_anormal,
                                          batch_logger=batch_logger,
                                          epoch_logger=epoch_logger,
                                          memory_bank=memory_bank)
                w, w_head, memory_bank, loss = local_model.update_weights(model=copy.deepcopy(model),
                                                                          model_head=copy.deepcopy(model_head),
                                                                          epoch=epoch)

                score, th, acc = test_inference(args, model, train_normal, valid_normal, valid_data, valid_labels,
                                                plot=False, file_path=args.result_folder+'contrastive')
                # save the model
                if acc > best_acc:

                    states = {'state_dict': model.state_dict()}
                    torch.save(states, checkpoint_path)

                    states_head = {'state_dict': model_head.state_dict()}
                    torch.save(states_head, head_checkpoint_path)

                    best_acc = acc

    elif args.mode is 'test':
        for c in range(args.num_users):
            print(f'\n------------------Client {c}: Test after completion of training --------------------')
            model = generate_model(args, input_size=all_data.train_data.shape[1])
            resume_path = os.path.join(args.checkpoint_folder, f'client{c}_best_model_{args.model_type}.pth')
            resume_checkpoint = torch.load(resume_path)
            model.load_state_dict(resume_checkpoint['state_dict'])

            score, th, acc = test_inference(args, model, train_normal, valid_normal, test_data, test_labels,
                                            plot=True, file_path=f'{args.result_folder}client{c}_contrastive')
            per_class_acc(all_data.y_test_multi_class, score, th)


# def parse_args():
#     parser = argparse.ArgumentParser(description='DAD training on Videos')
#
#     parser.add_argument('--mode', default='test', type=str, help='train | test(validation)')
#     parser.add_argument('--dataset', default='nsl', type=str, help='nsl | cicids')
#     parser.add_argument('--num_users', default=10, type=int, help='number of users in the FL system')
#     parser.add_argument('--frac', default=1, type=float, help='fraction of users participating in the learning')
#
#     parser.add_argument("--data_partition_type", help="whether it is a binary classification (normal or attack)",
#                         default='normalOverAll', choices=["normalOverAll", "DoS", "Probe", "U2R", "R2L"], type=str)
#     parser.add_argument('--data_distribution', default='iid', type=str, choices=['iid', 'non-iid', 'attack-split'])
#
#     parser.add_argument('--root_path', default='', type=str, help='root path of the dataset')
#     parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
#     parser.add_argument('--latent_dim', default=64, type=int, help='contrastive learning dimension')
#     parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')
#
#     parser.add_argument('--model_type', default='mlp', type=str, help='so far only resnet')
#     parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
#
#     parser.add_argument('--epochs', default=20, type=int, help='Number of total epochs to run')
#     parser.add_argument('--n_train_batch_size', default=5, type=int, help='Batch Size for normal training data')
#     parser.add_argument('--a_train_batch_size', default=200, type=int, help='Batch Size for anormal training data')
#     parser.add_argument('--val_batch_size', default=25, type=int, help='Batch Size for validation data')
#     parser.add_argument('--learning_rate', default=0.001, type=float,
#                         help='Initial learning rate (divided by 10 while training by lr scheduler)')
#     parser.add_argument('--lr_decay', default=100, type=int,
#                         help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
#     parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
#     parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
#     parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
#     parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
#     parser.add_argument('--tracking', default=True, type=ast.literal_eval,
#                         help='If true, BN uses tracking running stats')
#     parser.add_argument('--cal_vec_batch_size', default=20, type=int,
#                         help='batch size for calculating normal driving average vector.')
#
#     parser.add_argument('--tau', default=0.03, type=float,
#                         help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
#     parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
#     parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
#     parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
#     parser.set_defaults(nesterov=False)
#
#     parser.add_argument('--checkpoint_folder', default='./checkpoints/distributed/', type=str, help='folder to store checkpoints')
#     parser.add_argument('--result_folder', default='./result/distributed/', type=str, help='folder_to_store_results')
#     parser.add_argument('--log_folder', default='./logs/distributed/', type=str, help='folder to store log files')
#     parser.add_argument('--log_resume', default=False, type=ast.literal_eval, help='True|False: a flag controlling whether to create a new log file')
#     parser.add_argument('--normvec_folder', default='./normvec/distributed/', type=str, help='folder to store norm vectors')
#     parser.add_argument('--score_folder', default='./result/distributed/score/', type=str, help='folder to store scores')
#
#     parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
#     parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
#     parser.add_argument('--width_mult', default=2.0, type=float,
#                         help='hyper-parameters when using shufflenet|mobilenet')
#
#     parser.add_argument('--val_step', default=10, type=int, help='validate per val_step epochs')
#     parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')
#     parser.add_argument('--n_split_ratio', default=1.0, type=float,
#                         help='the ratio of normal driving samples will be used during training')
#     parser.add_argument('--a_split_ratio', default=1.0, type=float,
#                         help='the ratio of normal driving samples will be used during training')
#     parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')
#
#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    gv.init('distributed')
    args = gv.args

    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)

    args.mode = 'test'
    args.data_partition_type = 'normalOverAll'
    args.epochs = 1
    args.val_step = 10
    args.save_step = 10

    main(args)

