import torch
import torch.backends.cudnn as cudnn

import os
import csv
import random
import numpy as np
import global_vars as gv
from tqdm import tqdm
import copy
import time

from utils.data_split import get_dataset
from utils.utils import Logger, per_class_acc, split_evaluate, set_random_seed, AverageMeter
from utils.setup_NSL_2 import NSLKDD, NSLData
from model import generate_model
from models import mlp
from detection_two_classes import early_stop

from train_local_model import LocalUpdate
from utils.federated_utils import average_weights, test_inference
from nce_average import NCEAverage
from nce_criteria import NCECriterion


def training_loss(args, model, model_head, train_normal, train_anormal):
    losses = AverageMeter()
    model.eval()
    model_head.eval()
    train_anormal_loader = torch.utils.data.DataLoader(
        train_anormal,
        batch_size=args.a_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    train_normal_loader = torch.utils.data.DataLoader(
        train_normal,
        batch_size=args.n_train_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=True,
    )
    nce_average = NCEAverage(args.feature_dim, len(train_anormal),
                             len(train_normal), args.tau, args.Z_momentum)
    criterion = NCECriterion(len(train_anormal))

    for batch, ((normal_data, idx_n), (anormal_data, idx_a)) in enumerate(
            zip(train_normal_loader, train_anormal_loader)):
        if normal_data.size(0) != args.n_train_batch_size:
            break
        data = torch.cat((normal_data, anormal_data), dim=0)  # n_vec as well as a_vec are all normalized value
        if args.use_cuda:
            data = data.cuda()
            idx_a = idx_a.cuda()
            idx_n = idx_n.cuda()

        # ================forward====================
        unnormed_vec, normed_vec = model(data.float())
        vec = model_head(unnormed_vec)
        n_vec = vec[0:args.n_train_batch_size]
        a_vec = vec[args.n_train_batch_size:]
        outs, probs = nce_average(n_vec, a_vec, idx_n, idx_a,
                                       normed_vec[0:args.n_train_batch_size])
        loss = criterion(outs)
        losses.update(loss.item(), outs.size(0))

    print(losses.avg)
    return losses.avg


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

    # BUILD MODEL
    model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
    if args.use_cuda:
        model_head.cuda()
    global_model = generate_model(args, input_size=all_data.train_data.shape[1])

    # initialization
    valid_acc_ls = []
    moving_average = 4
    best_valid_acc = 0
    train_time = 0

    memory_bank_avg = []

    train_loss, train_accuracy = [], []
    rounds = int(args.epochs / args.local_epochs)

    # setup logger
    batch_logger = Logger(os.path.join(args.log_folder, f'batch.log'),
                           ['epoch', 'batch', 'loss', 'probs', 'lr'], args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, f'epoch_{args.data_distribution}.log'),
                           ['round', 'epoch', 'loss', 'probs', 'lr'], args.log_resume)

    checkpoint_path = os.path.join(args.checkpoint_folder,
                                   f'best_model_{args.model_type}.pth')
    head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                        f'best_model_{args.model_type}_head.pth')
    save_name = f'{args.data_distribution}_lr{args.learning_rate}_clients{args.num_users}_seed{args.manual_seed}_epochs{args.epochs}_le{args.local_epochs}_frac{args.frac}'

    if args.mode is 'train':
        for r in tqdm(range(rounds)):
            local_weights, local_head_weights, local_losses = [], [], []

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = rng.choice(range(args.num_users), m, replace=False)
            memory_bank_temp = []

            for num, idx in enumerate(idxs_users):
                print(f'\nGlobal Training Round : {r + 1} |--------({num}/{len(idxs_users)})'
                      + f'Clients Completed, Local Training of Client {idx} Starts!! ')
                start = time.time()
                local_model = LocalUpdate(args, idxs_normal=user_groups_normal[idx],
                                          idxs_anormal=user_groups_anormal[idx],
                                          dataset_normal=train_normal,
                                          dataset_anormal=train_anormal,
                                          batch_logger=batch_logger,
                                          epoch_logger=epoch_logger,
                                          memory_bank=memory_bank_avg)
                w, w_head, memory_bank, loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                                          model_head=copy.deepcopy(model_head),
                                                                          training_round=r,
                                                                          epoch_num=args.local_epochs)
                elapsed_time = time.time() - start
                print('Time elapse: ', elapsed_time)
                train_time += elapsed_time

                selected = int(args.memory_bank_size/m)
                memory_bank_temp.extend(memory_bank[-selected:])

                local_weights.append(copy.deepcopy(w))
                local_head_weights.append(copy.deepcopy(w_head))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            global_weights = average_weights(local_weights)
            global_head_weights = average_weights(local_head_weights)

            # update memory bank
            idx = list(range(len(memory_bank_temp)))
            rng.shuffle(idx)
            memory_bank_avg = [memory_bank_temp[i] for i in idx]

            # update global weights
            global_model.load_state_dict(global_weights)
            model_head.load_state_dict(global_head_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)
            print('average train loss', train_loss)

            # Calculate avg validation accuracy over all users at every round
            print(f'\nValidation Accuracy After {r + 1} Rounds of Federated Learning:')
            score, th = test_inference(args, global_model, train_normal, valid_normal, valid_data)
            _, acc_current, auc_current = split_evaluate(valid_labels, score, manual_th=th)

            # decide whether to stop the train
            # valid_acc_ls.append(auc_current)
            # if r+1 >= moving_average:
            #     if early_stop(valid_acc_ls, moving_average):
            #         print('!!!!!!!!!!!!!!!!!!! stop the current training as the performance degrades!!!!')
            #         break

            states = {'state_dict': global_model.state_dict()}
            torch.save(states, checkpoint_path)

            states_head = {'state_dict': model_head.state_dict()}
            torch.save(states_head, head_checkpoint_path)

            print(f'\n---------------------Test inference after {r+1} rounds of training -----------------------')
            performance_dict = dict()
            performance_dict['round'] = r+1
            performance_dict['training_loss'] = loss_avg
            score, th = test_inference(args, global_model, train_normal, valid_normal, test_data)

            split_evaluate(test_labels, score,
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
                lr = args.learning_rate * 0.5
                args.learning_rate = lr
                print(f'New learning rate: {lr}')

        print('the average time of per user processing is ', train_time/((r+1) * args.num_users * args.frac))

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
    args.num_users = 60
    args.local_epochs = 4
    args.epochs = 52
    args.frac = 0.1
    args.lr_decay = 80

    args.mode = 'train'
    main(args)


