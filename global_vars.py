import os
import ast
import argparse


def init(learning_type='fl'):
    parser = argparse.ArgumentParser(description='DAD training on Videos')

    parser.add_argument('--mode', default='test', type=str, help='train | test(validation)')
    parser.add_argument('--dataset', default='nsl', type=str, help='nsl | cicids')
    parser.add_argument('--num_users', default=10, type=int, help='number of users in the FL system')
    parser.add_argument('--frac', default=1, type=float, help='fraction of users participating in the learning')

    parser.add_argument("--data_partition_type", help="whether it is a binary classification (normal or attack)",
                        default='normalOverAll', choices=["normalOverAll", "DoS", "Probe", "U2R", "R2L"], type=str)
    parser.add_argument('--data_distribution', default='iid', type=str, choices=['iid', 'non-iid', 'attack-split'])

    parser.add_argument('--root_path', default='', type=str, help='root path of the dataset')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--latent_dim', default=64, type=int, help='contrastive learning dimension')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')

    parser.add_argument('--model_type', default='mlp', type=str, help='so far only resnet')
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
    parser.add_argument('--result_folder', default='./result/', type=str, help='folder_to_store_results')
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

    global args
    args = parser.parse_args()
    print(args)

    # the folder path
    args.checkpoint_folder = args.checkpoint_folder + learning_type + '/'
    args.result_folder = args.result_folder + learning_type + '/'
    args.log_folder = args.log_folder + learning_type + '/'
    args.normvec_folder = args.normvec_folder + learning_type + '/'
    args.score_folder = args.score_folder + learning_type + '/'

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.normvec_folder):
        os.makedirs(args.normvec_folder)
    if not os.path.exists(args.score_folder):
        os.makedirs(args.score_folder)