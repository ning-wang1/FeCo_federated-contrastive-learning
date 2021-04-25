import torch
import torch.backends.cudnn as cudnn

import os
import numpy as np
import argparse
from tqdm import tqdm
import copy

from utils.data_split import get_dataset


from test import get_normal_vector, split_acc_diff_threshold, cal_score, cal_score_downstream,\
    get_new_represent
from utils.utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize,\
    post_process, evaluate, get_score
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.setup_NSL import NSL_KDD, NSL_data
from model import generate_model
from models import mlp
# from models import resnet, shufflenet, shufflenetv2, mobilenet, mobilenetv2
import ast
from utils.utils import split_evaluate
from train_local_model import LocalUpdate
from utils.federated_utils import average_weights, test_inference


def parse_args():
    parser = argparse.ArgumentParser(description='DAD training on Videos')

    parser.add_argument('--dataset', default='nsl', type=str, help='nsl | cicids')
    parser.add_argument('--num_users', default=10, type=int, help='number of users in the FL system')
    parser.add_argument('--frac', default=1, type=float, help='fraction of users participating in the learning')

    parser.add_argument('--mode', default='test', type=str, help='train | test(validation)')
    parser.add_argument('--root_path', default='', type=str, help='root path of the dataset')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--latent_dim', default=64, type=int, help='contrastive learning dimension')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')

    parser.add_argument('--model_type', default='mlp', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')

    parser.add_argument('--epochs', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_train_batch_size', default=5, type=int, help='Batch Size for normal training data')
    parser.add_argument('--a_train_batch_size', default=200, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=25, type=int, help='Batch Size for validation data')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--n_threads', default=8, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--cal_vec_batch_size', default=20, type=int,
                        help='batch size for calculating normal driving average vector.')

    parser.add_argument('--tau', default=0.03, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)

    parser.add_argument('--checkpoint_folder', default='./checkpoints/', type=str, help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval, help='True|False: a flag controlling whether to create a new log file')
    parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    parser.add_argument('--score_folder', default='./result/score/', type=str, help='folder to store scores')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
    parser.add_argument('--width_mult', default=2.0, type=float,
                        help='hyper-parameters when using shufflenet|mobilenet')
    parser.add_argument('--val_step', default=10, type=int, help='validate per val_step epochs')
    parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')
    parser.add_argument('--n_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--a_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # load dataset and user group
    train_normal, train_anormal, valid_data, test_data, user_groups_normal, user_groups_anormal = get_dataset(args)
    _, test_labels = test_data[:]
    _, valid_labels = valid_data[:]

    len_neg = train_anormal.__len__()
    len_pos = train_normal.__len__()

    # BUILD MODEL
    model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
    if args.use_cuda:
        model_head.cuda()
    global_model = generate_model(args)

    begin_epoch = 1
    best_acc = 0
    valid_auc = 0
    memory_bank = [[] for i in range(args.num_users)]

    # setup logger
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                        ['epoch', 'accuracy', 'normal_acc', 'anormal_acc', 'threshold', 'acc_list',
                         'normal_acc_list', 'anormal_acc_list'], args.log_resume)

    # copy weights
    global_weights = global_model.state_dict()

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_head_weights, local_losses = [], [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            print('\n-------------------- Local Training of Client {} Starts!! ----------------------'.format(idx))
            local_model = LocalUpdate(args, idxs_normal=user_groups_normal[idx],
                                      idxs_anormal=user_groups_anormal[idx],
                                      dataset_normal=train_normal,
                                      dataset_anormal=train_anormal,
                                      batch_logger=batch_logger,
                                      epoch_logger=epoch_logger,
                                      memory_bank=memory_bank[idx])
            w, w_head, memory_bank[idx], loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                                           model_head=copy.deepcopy(model_head),
                                                                           epoch=epoch)
            local_weights.append(copy.deepcopy(w))
            local_head_weights.append(copy.deepcopy(w_head))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        global_head_weights = average_weights(local_head_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        model_head.load_state_dict(global_head_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            print(f'\nEvaluating Performance of Client {c}')
            local_model = LocalUpdate(args, idxs_normal=user_groups_normal[c],
                                      idxs_anormal=user_groups_anormal[c],
                                      dataset_normal=train_normal,
                                      dataset_anormal=train_anormal,
                                      batch_logger=batch_logger,
                                      epoch_logger=epoch_logger,
                                      memory_bank=memory_bank[c])

            acc = local_model.inference(model=global_model)
            list_acc.append(acc)

        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        print(f'\nValidation Accuracy after {epoch + 1} global rounds:')
        valid_auc_current = test_inference(args, global_model, train_normal, valid_data, valid_labels)
        if valid_auc_current - valid_auc < -0.002:
            print('>>>>>>>>>>>>>>>>> stop the current training as the performance degrades!!!!')
            break
        else:
            valid_acc = valid_auc_current

    print('\n------------------Test inference after completion of training --------------------')
    test_inference(args, global_model, train_normal, test_data, test_labels, plot=True)
