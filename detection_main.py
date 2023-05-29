import torch
import torch.backends.cudnn as cudnn

import os
import csv
import numpy as np
import global_vars as gv

from utils.utils import adjust_learning_rate, AverageMeter, Logger, l2_normalize, get_threshold
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.setup_NSL_2 import NSLKDD, NSLData
from model import generate_model
from models import mlp
from utils.utils import split_evaluate, per_class_acc, set_random_seed


def get_normal_vec(dataloader, cal_vec_batch_size):
    total_batch = int(len(dataloader))

    for batch, (normal_data, idx) in enumerate(dataloader):
        if batch == 0:
            normal_vec = torch.zeros((1, normal_data.shape[1])).detach()
        normal_data = normal_data.detach()
        normal_vec = (torch.sum(normal_data, dim=0) + normal_vec * batch * cal_vec_batch_size) / (
                (batch + 1) * cal_vec_batch_size)
        if (batch + 1) % 1000 == 0:
            print(f'Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}')

    normal_vec = l2_normalize(normal_vec).float()
    normal_vec.reshape([1, -1])
    return normal_vec


def cal_score(normal_vec, test_loader, score_folder=None):
    """
    Generate and save scores
    """
    total_batch = int(len(test_loader))
    sim_1_list = torch.zeros(0)
    label_list = torch.zeros(0).type(torch.LongTensor)

    for batch, data1 in enumerate(test_loader):
        out_1 = data1[0].float().detach()
        sim_1 = torch.mm(out_1, normal_vec.t())

        label_list = torch.cat((label_list, data1[1].squeeze().cpu()))
        sim_1_list = torch.cat((sim_1_list, sim_1.squeeze().cpu()))
        if (batch + 1) % 100 == 0:
            print(f'Calculating Scores---- Evaluating: Batch {batch + 1} / {total_batch}')
    if score_folder is not None:
        np.save(score_folder, sim_1_list.numpy())
        print('score.npy is saved')
    return sim_1_list.numpy()


def main(args, fea_selection):
    rng = set_random_seed(args.manual_seed, args.use_cuda)
    if args.nesterov:
        dampening = 1
    else:
        dampening = args.dampening

    attack_type = {'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}

    if args.data_partition_type is "normalOverAll":
        all_data = NSLKDD(rng, data_type=None, fea_selection=fea_selection)
        normal_data = NSLKDD(rng, data_type='normal', fea_selection=fea_selection)

    else:
        attack = [(args.data_partition_type, attack_type[args.data_partition_type])]
        all_data = NSLKDD(rng, attack, data_type=None, fea_selection=fea_selection)
        normal_data = NSLKDD(rng, attack, data_type='normal', fea_selection=fea_selection)

    if args.mode == 'train':
        print("=====================================Generating Model=========================================")

    elif args.mode == 'test':

        print("================================ Loading Normal Data =====================================")
        training_normal_data = NSLData(normal_data.train_data, normal_data.train_labels)
        training_normal_size = int(len(training_normal_data) * args.n_split_ratio)
        training_normal_data = torch.utils.data.Subset(training_normal_data, np.arange(training_normal_size))

        train_normal_loader_for_test = torch.utils.data.DataLoader(
            training_normal_data,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'train_normal_loader_for_test (size: {len(training_normal_data)})')

        validation_data = NSLData(normal_data.validation_data, normal_data.validation_labels)
        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("================================ Loading Test Data =====================================")
        test_data = NSLData(all_data.test_data, all_data.test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("=================================START EVALUATING=========================================")
        normal_vec = get_normal_vec(train_normal_loader_for_test, args.cal_vec_batch_size)
        np.save(os.path.join(args.normvec_folder, 'normal_vec.npy'), normal_vec.cpu().numpy())

        # compute a decision-making threshold using the validation dataset
        valid_scores = cal_score(normal_vec, validation_loader)
        th = get_threshold(valid_scores, percent=5)
        print(f'the threshold is set as {th}')

        # evaluating the scores of the test dataset and show the IDS performance
        score_folder = args.score_folder
        score = cal_score(normal_vec, test_loader, score_folder)
        # score = get_score(score_folder)

        # split_evaluate_two_st='eps(pred_consist, all_data.test_labels, score, manual_th=th, perform_dict=None)

        performance_dict = dict()
        split_evaluate(all_data.test_labels, score, plot=True,
                       filename=f'{args.result_folder}contrastive', manual_th=th, perform_dict=performance_dict)
        per_class_acc(all_data.y_test_multi_class, score, th, perform_dict=performance_dict)

        with open(f'{args.score_folder}metrics{args.manual_seed}.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
            writer.writeheader()
            writer.writerow(performance_dict)


if __name__ == '__main__':
    gv.init('centralized')
    args = gv.args

    args.data_partition_type = 'normalOverAll'

    if args.data_partition_type is 'normalOverAll':
        # args.epochs = 60
        # args.val_step = 30
        # args.save_step = 30
        # args.tau = 0.03
        # args.learning_rate = 0.001
        # args.lr_decay = 45
        # args.n_train_batch_size = 5

        args.epochs = 60
        args.val_step = 60
        args.save_step = 60
        args.tau = 0.02
        args.learning_rate = 0.001
        args.lr_decay = 45
        args.n_train_batch_size =5

    else:
        args.epochs = 50
        args.val_step = 10
        args.save_step = 10
    #
    args.manual_seed = 1
    args.mode = 'train'
    main(args, fea_selection=True)
    args.mode = 'test'
    main(args, fea_selection=True)

