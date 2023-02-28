import torch

import os
import csv
import random
import numpy as np
import copy
import global_vars as gv

from utils.data_split import get_dataset
from utils.utils import Logger, per_class_acc, split_evaluate, set_random_seed
from utils.setup_NSL_2 import NSLKDD, NSLData
from model import generate_model
from models import mlp

from train_local_model import LocalUpdate
from utils.federated_utils import average_weights, test_inference


def main(args):
    rng = set_random_seed(args.manual_seed, args.use_cuda)
    # load dataset and user group
    if args.dataset == 'nsl':
        attack_type = {'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}

        if args.data_partition_type is "normalOverAll":
            all_data = NSLKDD(rng, data_type=None)
            normal_data = NSLKDD(rng, data_type='normal')
            anormal_data = NSLKDD(rng, data_type='anomaly')
        else:
            attack = [(args.data_partition_type, attack_type[args.data_partition_type])]
            all_data = NSLKDD(rng, attack, data_type=None)
            normal_data = NSLKDD(rng, attack, data_type='normal')
            anormal_data = NSLKDD(rng, attack, data_type='anomaly')

    train_normal, train_anormal, valid_data, test_data, user_groups_normal, user_groups_anormal = get_dataset(
        args, all_data, normal_data, anormal_data)

    _, test_labels = test_data[:]
    _, valid_labels = valid_data[:]
    valid_normal = NSLData(normal_data.validation_data, normal_data.validation_labels)

    # setup logger
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    save_name = f'{args.data_distribution}_lr{args.learning_rate}_clients{args.num_users}_seed{args.manual_seed}'

    if args.mode is 'train':
        for c in range(args.num_users):
            print(f'\n-------------------- Local Training of Client {c} Starts!! ----------------------')
            # BUILD MODEL
            model = generate_model(args, input_size=all_data.train_data.shape[1])
            model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
            if args.use_cuda:
                model_head.cuda()

            # initial
            memory_bank = []
            checkpoint_path = os.path.join(args.checkpoint_folder,
                                           f'client{c}_best_model_{args.model_type}.pth')
            head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                f'client{c}_best_model_{args.model_type}_head.pth')

            local_model = LocalUpdate(args, idxs_normal=user_groups_normal[c],
                                      idxs_anormal=user_groups_anormal[c],
                                      dataset_normal=train_normal,
                                      dataset_anormal=train_anormal,
                                      batch_logger=batch_logger,
                                      epoch_logger=epoch_logger,
                                      memory_bank=memory_bank)

            w, w_head, memory_bank, loss = local_model.update_weights(model=copy.deepcopy(model),
                                                                      model_head=copy.deepcopy(model_head),
                                                                      training_round=1,
                                                                      epoch_num=args.local_epochs)
            model.load_state_dict(w)
            model_head.load_state_dict(w_head)

            states = {'state_dict': model.state_dict()}
            torch.save(states, checkpoint_path)

            states_head = {'state_dict': model_head.state_dict()}
            torch.save(states_head, head_checkpoint_path)

            print(f'\n---------------------Test inference  -----------------------')
            performance_dict = dict()
            performance_dict['client_id'] = c
            score, th = test_inference(args, model, train_normal, valid_normal, test_data)

            split_evaluate(test_labels, score, plot=True,
                           filename=f'{args.result_folder}contrastive_c{c}'+save_name,
                           manual_th=th,
                           perform_dict=performance_dict)
            per_class_acc(all_data.y_test_multi_class, score, th, perform_dict=performance_dict)
            if c == 0:
                with open(f'{args.score_folder}metrics_' + save_name + '.csv', 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
                    writer.writeheader()
                    writer.writerow(performance_dict)
            else:
                with open(f'{args.score_folder}metrics_' + save_name + '.csv', 'a+') as f:
                    writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
                    writer.writerow(performance_dict)

    elif args.mode is 'test':
        for c in range(args.num_users):
            print(f'\n------------------Client {c}: Test after completion of training --------------------')
            model = generate_model(args, input_size=all_data.train_data.shape[1])
            resume_path = os.path.join(args.checkpoint_folder, f'client{c}_best_model_{args.model_type}.pth')
            resume_checkpoint = torch.load(resume_path)
            model.load_state_dict(resume_checkpoint['state_dict'])

            score, th = test_inference(args, model, train_normal, valid_normal, test_data)
            per_class_acc(all_data.y_test_multi_class, score, th)


if __name__ == "__main__":
    gv.init('distributed')
    args = gv.args

    args.data_partition_type = 'normalOverAll'
    args.data_distribution = 'iid'
    args.learning_rate = 0.001
    args.local_epochs = 30
    args.num_users = 60

    args.cal_vec_batch_size = 50

    args.mode = 'train'
    main(args)

