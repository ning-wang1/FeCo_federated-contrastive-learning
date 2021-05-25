import torch
import torch.backends.cudnn as cudnn

import os
import ast
import random
import numpy as np
import argparse
import global_vars as gv
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

    # BUILD MODEL
    model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
    if args.use_cuda:
        model_head.cuda()
    global_model = generate_model(args, input_size=all_data.train_data.shape[1])

    valid_acc_best = 0
    memory_bank = [[] for i in range(args.num_users)]

    # setup logger
    batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                          args.log_resume)

    checkpoint_path = os.path.join(args.checkpoint_folder,
                                   f'best_model_{args.model_type}.pth')
    head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                        f'best_model_{args.model_type}_head.pth')

    train_loss, train_accuracy = [], []
    print_every = 2

    if args.mode is 'train':
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
            _, _, acc_current = test_inference(args, global_model, train_normal, valid_normal,
                                               valid_data, valid_labels,
                                               plot=False, file_path=args.result_folder+'contrastive')
            if acc_current < valid_acc_best:
                if acc_current < valid_acc_best - 0.01:
                    print('>>>>>>>>>>>>>>>>> stop the current training as the performance degrades!!!!')
                    break
            else:
                states = {'state_dict': global_model.state_dict()}
                torch.save(states, checkpoint_path)

                states_head = {'state_dict': model_head.state_dict()}
                torch.save(states_head, head_checkpoint_path)

        print('\n------------------Test inference after completion of training --------------------')
        score, th, acc = test_inference(args, global_model, train_normal, valid_normal, test_data, test_labels,
                                        plot=True, file_path=args.result_folder+'contrastive')
        per_class_acc(all_data.y_test_multi_class, score, th)

    elif args.mode is 'test':
        model = generate_model(args, input_size=all_data.train_data.shape[1])
        resume_path = os.path.join(args.checkpoint_folder, f'best_model_{args.model_type}.pth')
        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])

        score, th, acc = test_inference(args, model, train_normal, valid_normal, test_data, test_labels,
                                        plot=True, file_path=args.result_folder+'contrastive')
        per_class_acc(all_data.y_test_multi_class, score, th)


if __name__ == "__main__":
    gv.init('fl')
    args = gv.args

    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)

    args.mode = 'test'
    args.data_partition_type = 'normalOverAll'
    args.epochs = 2
    args.val_step = 10
    args.save_step = 10

    main(args)

