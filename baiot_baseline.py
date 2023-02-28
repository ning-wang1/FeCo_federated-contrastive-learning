import os
import argparse
import csv
import numpy as np
import random
import global_vars as gv
import torch
import torch.backends.cudnn as cudnn


from models import mlp
from model import generate_model
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from centralized_main import train
from utils.setup_BaIoT import BaIoT, BaIoT_data, DEVICE_NAMES
from models.vae import NSL_MLP_Autoencoder, AETrainer
from utils.utils import split_evaluate, adjust_learning_rate,\
    Logger, l2_normalize, get_score, get_threshold, split_evaluate_w_label
from test import get_normal_vector, split_acc_diff_threshold, cal_score, save_score_with_label
from models.iso_forest import IsoForest
from utils import classifier as clf
from sklearn.neighbors import LocalOutlierFactor
from models.ae import AETrainer, Autoencoder


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--xp_dir", default='./result/one class/baiot/', help="directory for the experiment", type=str)
    # parser.add_argument("--plot_dir", default='./result/one class/detection', help="directory for figures", type=str)
    parser.add_argument("--model_name", type=str, default='iso_forest',
                        choices=["isoForest", 'kde', 'svm', 'ae'])

    # for IsoForest
    parser.add_argument("--n_estimators", help="Specify the number of base estimators in the ensemble",
                        type=int, default=64)

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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for Deep SVDD network training. Default=0.001')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr_milestone', default=[500],
                        help='Lr scheduler milestones at which lr is multiplied by 0.1.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 penalty) hyperparameter')
    parser.add_argument('--n_jobs_dataloader', type=int, default=0, help='Number of workers for data loading.'
                        ' 0 means that the data will be loaded in the main process.')

    args = parser.parse_args()
    return args


def main(device_name, args):
    # args.model_name = 'ae'
    # args.model_name = 'svm'
    print('Options: {}', args)

    data = BaIoT(device_name)
    x_train_attack, x_train_normal = data.x_train_attack, data.x_train_normal
    x_val_normal = data.x_val_normal
    # x_normal = np.concatenate([x_train_normal, x_val_normal], axis=0)
    x_val, y_val, x_test, y_test = data.x_val, data.y_val, data.x_test, data.y_test

    plot_save_path = os.path.join(args.xp_dir, device_name + '_' + args.model_name)

    if args.model_name is 'isoForest':
        # initialize Isolation Forest
        model = IsoForest(args.seed, train_data=x_train_normal, test_data=x_test,
                          test_labels=y_test, n_estimators=args.n_estimators)

        # train model and predict
        model.train()
        model.predict(save_path=plot_save_path)
    elif args.model_name is 'svm':
        # subsampleing the data to make the training faster
        n_record = len(data.x_train)
        idx = np.arange(n_record)
        np.random.shuffle(idx)
        # MLP n_records, svm 2000,lgr 10000, bnb n_records, KNN n_records, DTC n_records
        # n_select = 10000
        n_select = n_record
        # train other models including SVM LG BNB DT
        model = clf.classifier('SVM', data.x_train[idx[:n_select]], data.y_train[idx[:n_select]])

        # y_pred_score = model.predict_proba(x_test)
        y_pred_score = model.decision_function(x_test)
        y_pred = model.predict(x_test)

        # save the results
        dic = dict()
        _, _, auc = split_evaluate(y_test, y_pred_score, plot=True,
                                   filename=plot_save_path, perform_dict=dic)
        dic['auc'] = auc

        # split_evaluate_w_label(y_test, y_pred)

        with open(plot_save_path + '.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=dic.keys())
            writer.writeheader()
            writer.writerow(dic)

    elif args.model_name is 'lof':

        model = LocalOutlierFactor(n_neighbors=20)
        # y_pred = model.fit_predict(x_test)
        # y_pred_scores = model.negative_outlier_factor_

        n = x_test.shape[0]
        idx = list(range(n))
        np.random.shuffle(idx)
        n_split = 40
        samples = int(n/n_split)

        test_samples = x_test[0: samples, ]
        model.fit_predict(test_samples)
        y_pred_scores = model.negative_outlier_factor_

        for i in range(1, n_split):
            print(i)
            if i == n_split - 1:
                test_samples = x_test[i * samples:, ]
            else:
                test_samples = x_test[i * samples:(i + 1) * samples, ]

            model.fit_predict(test_samples)
            y_pred_score = model.negative_outlier_factor_
            y_pred_scores = np.concatenate([y_pred_scores, y_pred_score])

        # save the results
        dic = dict()
        _, _, auc = split_evaluate(y_test, y_pred_scores, plot=True,
                                   filename=plot_save_path, perform_dict=dic)
        dic['auc'] = auc

        with open(plot_save_path + '.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=dic.keys())
            writer.writeheader()
            writer.writerow(dic)

    elif args.model_name is 'ae':
        device = args.device
        if not torch.cuda.is_available():
            device = 'cpu'

        # train autoencoder on dataset
        model = Autoencoder()
        ae_trainer = AETrainer(args.optimizer_name, lr=args.lr, n_epochs=args.n_epochs,
                               lr_milestones=args.lr_milestone, batch_size=args.batch_size,
                               weight_decay=args.weight_decay, device=device,
                               n_jobs_dataloader=args.n_jobs_dataloader)
        model = ae_trainer.train(data, model)
        ae_trainer.test(data, model, file_path=args.xp_dir)


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device_name = DEVICE_NAMES[4]
    print(device_name)
    args = get_args()
    args.seed = seed
    args.n_epochs = 10

    args.model_name = 'isoForest'
    main(device_name, args)