import torch
import numpy as np
import random
import argparse
import logging
import os

from utils.setup_NSL import NSL_KDD
from models.iso_forest import IsoForest
from models.kde import KDE
from models.ocsvm import SVM
from models.vae import NSL_MLP_Autoencoder, AETrainer
from utils.setup_NSL import NSL_Dataset
from utils.utils import get_threshold
# from utils.logs import log_exp_config, log_isoForest, log_AD_results


# ====================================================================
# Parse arguments
# --------------------------------------------------------------------
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--dataset", help="dataset name", type=str, choices=["mnist", "cifar10", "gtsrb"])
    parser.add_argument("--data_partition_type", help="whether it is a binary classification (normal or attack)",
                        default='normalOverAll', choices=["normalOverAll", "DoS", "Probe", "U2R", "R2L"], type=str)
    parser.add_argument("--xp_dir", default='./result', help="directory for the experiment", type=str)
    parser.add_argument("--plot_dir", default='./result/detection', help="directory for figures", type=str)
    parser.add_argument("--model_name", type=str, default='iso_forest',
                        choices=["isoForest", 'kde', 'svm', 'ae'])

    # for IsoForest
    parser.add_argument("--n_estimators", help="Specify the number of base estimators in the ensemble",
                        type=int, default=100)
    parser.add_argument("--max_samples", help="Number of samples drawn to train each base estimator",
                        type=int, default=256)
    parser.add_argument("--contamination", help="Expected fraction of outliers in the training set (contamination)",
                        type=float, default=0.1)

    # for kde
    parser.add_argument("--kde_kernel", help="kernel", type=str, default='gaussian',
                        choices=["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])
    parser.add_argument("--kde_GridSearchCV", help="Use GridSearchCV to determine bandwidth",
                        type=int, default=0)

    # for svm
    parser.add_argument("--loss", help="loss function", default="OneClassSVM",
                        type=str, choices=["OneClassSVM", "SVC"])
    parser.add_argument("--svm_kernel", help="kernel", type=str, default='rbf',
                        choices=["linear", "poly", "rbf", "sigmoid", "DegreeKernel", "WeightedDegreeKernel"])
    parser.add_argument("--svm_GridSearchCV", help="Use GridSearchCV to determine bandwidth",
                        type=int, default=0)

    # for autoencoder
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
    parser.add_argument('--optimizer_name', choices=(['adam', 'amsgrad']), default='adam',
                        help='Name of the optimizer to use for Deep SVDD network training.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate for Deep SVDD network training. Default=0.001')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr_milestone', default=[20],
                        help='Lr scheduler milestones at which lr is multiplied by 0.1.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for mini-batch training.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 penalty) hyperparameter')
    parser.add_argument('--n_jobs_dataloader', type=int, default=0, help='Number of workers for data loading.'
                        ' 0 means that the data will be loaded in the main process.')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    args.model_name = 'ae'
    args.seed = 4
    print('Options: {}', args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    attack_type = {'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}

    # load data
    if args.data_partition_type is "normalOverAll":
        data = NSL_KDD(data_type=None)
        normal_data = NSL_KDD(data_type='normal')
    else:
        attack = [(args.data_partition_type, attack_type[args.data_partition_type])]
        data = NSL_KDD(attack, data_type=None)
        normal_data = NSL_KDD(attack, data_type='normal')

    if not os.path.exists(args.xp_dir):
        os.mkdir(args.xp_dir)
    assert os.path.exists(args.xp_dir)
    plot_save_path = os.path.join(args.plot_dir, args.data_partition_type + '_' + args.model_name)

    # logging
    logging.basicConfig(level=logging.INFO)

    if args.model_name is 'isoForest':
        # initialize Isolation Forest
        model = IsoForest(args.seed, train_data=normal_data.train_data, test_data=data.test_data,
                          test_labels=data.test_labels, n_estimators=args.n_estimators,
                          max_samples=args.max_samples, contamination=args.contamination)
        # train model and predict
        model.train()
        model.predict(save_path=plot_save_path)

    elif args.model_name is 'kde':
        # initialize KDE
        model = KDE(train_data=normal_data.train_data, test_data=data.test_data,
                    test_labels=data.test_labels,
                    kernel=args.kde_kernel)

        # train KDE model and predict
        model.train(bandwidth_GridSearchCV=bool(args.kde_GridSearchCV))
        model.predict(save_path=plot_save_path)

    elif args.model_name is 'svm':
        # initialize OC-SVM
        model = SVM(loss=args.loss, normal_data=normal_data.train_data, data=data, kernel=args.svm_kernel)

        # train OC-SVM model
        model.train(GridSearch=args.svm_GridSearchCV)

        # predict scores
        model.predict(save_path=plot_save_path, which_set='test')

    elif args.model_name is 'ae':
        device = args.device
        if not torch.cuda.is_available():
            device = 'cpu'

        dataset = NSL_Dataset(normal_class=1, data_partition_type=args.data_partition_type)

        # train autoencoder on dataset
        model = NSL_MLP_Autoencoder()
        ae_trainer = AETrainer(args.optimizer_name, lr=args.lr, n_epochs=args.n_epochs,
                               lr_milestones=args.lr_milestone, batch_size=args.batch_size,
                               weight_decay=args.weight_decay, device=device,
                               n_jobs_dataloader=args.n_jobs_dataloader)
        model = ae_trainer.train(dataset, model)
        ae_trainer.test(dataset, model)

    else:
        print('error: the model name is not in file')

    # pickle/serialize
    # model.dump_model(filename=args.xp_dir + "/model.p")


if __name__ == '__main__':
    main()
