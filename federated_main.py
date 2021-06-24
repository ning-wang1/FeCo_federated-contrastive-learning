import torch
import torch.backends.cudnn as cudnn

import os
import csv
import random
import numpy as np
import global_vars as gv
from tqdm import tqdm
import copy

from utils.data_split import get_dataset
from utils.utils import Logger, per_class_acc, split_evaluate, set_random_seed
from utils.setup_NSL import NSL_KDD, NSL_data
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
            all_data = NSL_KDD(rng, data_type=None)
            normal_data = NSL_KDD(rng, data_type='normal')
            anormal_data = NSL_KDD(rng, data_type='anomaly')
        else:
            attack = [(args.data_partition_type, attack_type[args.data_partition_type])]
            all_data = NSL_KDD(rng, attack, data_type=None)
            normal_data = NSL_KDD(rng, attack, data_type='normal')
            anormal_data = NSL_KDD(rng, attack, data_type='anomaly')

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
    batch_logger = [Logger(os.path.join(args.log_folder, f'client{i}_batch.log'),
                           ['epoch', 'batch', 'loss', 'probs', 'lr'], args.log_resume)
                    for i in range(args.num_users)]
    epoch_logger = [Logger(os.path.join(args.log_folder, f'client{i}_epoch.log'),
                           ['round', 'epoch', 'loss', 'probs', 'lr'],
                    args.log_resume) for i in range(args.num_users)]

    checkpoint_path = os.path.join(args.checkpoint_folder,
                                   f'best_model_{args.model_type}.pth')
    head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                        f'best_model_{args.model_type}_head.pth')
    save_name = f'{args.data_distribution}_lr{args.learning_rate}_clients{args.num_users}_seed{args.manual_seed}_epochs{args.epochs}_le{args.local_epochs}_frac{args.frac}'

    train_loss, train_accuracy = [], []
    rounds = int(args.epochs/args.local_epochs)

    if args.mode is 'train':
        for r in tqdm(range(rounds)):
            local_weights, local_head_weights, local_losses = [], [], []

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for num, idx in enumerate(idxs_users):
                print(f'\nGlobal Training Round : {r + 1} |--------({num}/{len(idxs_users)})'
                      + f'Clients Completed, Local Training of Client {idx} Starts!! ')
                local_model = LocalUpdate(args, idxs_normal=user_groups_normal[idx],
                                          idxs_anormal=user_groups_anormal[idx],
                                          dataset_normal=train_normal,
                                          dataset_anormal=train_anormal,
                                          batch_logger=batch_logger[idx],
                                          epoch_logger=epoch_logger[idx],
                                          memory_bank=memory_bank[idx])
                w, w_head, memory_bank[idx], loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                                               model_head=copy.deepcopy(model_head),
                                                                               training_round=r,
                                                                               epoch_num=args.local_epochs)
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

            # Calculate avg validation accuracy over all users at every round
            print(f'\nValidation Accuracy After {r + 1} Rounds of Federated Learning:')
            score, th = test_inference(args, global_model, train_normal, valid_normal, valid_data)
            _, acc_current, _ = split_evaluate(valid_labels, score, manual_th=th)

            # decide whether to stop the train
            if acc_current < valid_acc_best:
                if acc_current < valid_acc_best - 0.01:
                    print('!!!!!!!!!!!!!!!!!!! stop the current training as the performance degrades!!!!')
                    break
            else:
                states = {'state_dict': global_model.state_dict()}
                torch.save(states, checkpoint_path)

                states_head = {'state_dict': model_head.state_dict()}
                torch.save(states_head, head_checkpoint_path)

            print(f'\n---------------------Test inference after {r+1} rounds of training -----------------------')
            performance_dict = dict()
            performance_dict['round'] = r+1
            score, th = test_inference(args, global_model, train_normal, valid_normal, test_data)

            split_evaluate(test_labels, score, plot=True,
                           filename=f'{args.result_folder}contrastive_r{r}' + save_name,
                           manual_th=th,
                           perform_dict=performance_dict)
            per_class_acc(all_data.y_test_multi_class, score, th, perform_dict=performance_dict)

            if r == 0:
                with open(f'{args.score_folder}metrics_' + save_name + '.csv', 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
                    writer.writeheader()
                    writer.writerow(performance_dict)
            else:
                with open(f'{args.score_folder}metrics_' + save_name + '.csv', 'a+') as f:
                    writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
                    writer.writerow(performance_dict)
            if (r+1) * args.local_epochs % args.lr_decay == 0:
                lr = args.learning_rate * (0.1 ** (r * args.local_epochs // args.lr_decay))
                args.learning_rate = lr
                print(f'New learning rate: {lr}')

    elif args.mode is 'test':
        model = generate_model(args, input_size=all_data.train_data.shape[1])
        resume_path = os.path.join(args.checkpoint_folder, f'best_model_{args.model_type}.pth')
        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])

        score, th = test_inference(args, model, train_normal, valid_normal, test_data)
        split_evaluate(test_labels, score, plot=True,
                       filename=f'{args.result_folder}contrastive_final',
                       manual_th=th)
        per_class_acc(all_data.y_test_multi_class, score, th)


if __name__ == "__main__":
    gv.init('fl')
    args = gv.args
    args.manual_seed = 1

    args.data_partition_type = 'normalOverAll'
    args.data_distribution = 'iid'
    args.learning_rate = 0.001
    args.num_users = 50
    args.local_epochs = 2
    args.epochs = 80
    args.frac = 0.1

    args.mode = 'train'
    main(args)
    args.mode = 'test'
    main(args)

