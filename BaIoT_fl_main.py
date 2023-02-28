import torch
import os
import csv
import copy
import time
import global_vars as gv
import numpy as np
from tqdm import tqdm
from models import mlp

from model import generate_model
from train_local_model import LocalUpdate
from utils.setup_BaIoT import BaIoT, BaIoT_data
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.data_split import get_dataset
from utils.utils import Logger, per_class_acc, split_evaluate, set_random_seed, AverageMeter
from utils.federated_utils import average_weights, test_inference
from federated_main import training_loss


def main(args, device_names):
    """
    train a FL model and test the model accuracy. Each device is one client, and these clients cooperatively
    train a global model.

    For training phase, after each round, test the accuracy of each device
    For testing phase, test the accuracy of the global model on the local data at each device
    """

    rng = set_random_seed(args.manual_seed, args.use_cuda)
    # load dataset and user group

    x_train_a_ls = []
    x_train_n_ls = []
    x_val_n_ls = []
    y_val_n_label_ls = []
    x_test_ls = []
    y_test_ls = []

    for device_name in device_names:
        data = BaIoT(device_name)
        x_train_attack = np.concatenate((data.x_train_attack, data.x_val_attack), axis=0)
        x_train_normal = data.x_train_normal
        x_val_normal = data.x_val_normal
        x_test, y_test = data.x_test, data.y_test

        x_train_a_ls.append(x_train_attack)
        x_train_n_ls.append(x_train_normal)
        x_val_n_ls.append(x_val_normal)
        y_val_n_label_ls.append(np.zeros(len(x_val_normal)))
        x_test_ls.append(x_test)
        y_test_ls.append(y_test)

    # BUILD MODEL
    model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
    if args.use_cuda:
        model_head.cuda()
    global_model = generate_model(args, input_size=x_train_attack.shape[1])

    # initialization
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
            idxs_users = list(range(args.num_users))
            memory_bank_temp = []

            for idx, num in enumerate(idxs_users):
                print(f'\nGlobal Training Round : {r + 1} |--------({num}/{len(idxs_users)})'
                      + f'Clients Completed, Local Training of Client {idx} Starts!! ')

                start = time.time()
                local_model = LocalUpdate(args, idxs_normal=np.arange(len(x_train_n_ls[idx])),
                                          idxs_anormal=np.arange(len(x_train_a_ls[idx])),
                                          dataset_normal=BaIoT_data(x_train_n_ls[idx], np.zeros(len(x_train_n_ls[idx]))),
                                          dataset_anormal=BaIoT_data(x_train_a_ls[idx], np.ones(len(x_train_a_ls[idx]))),
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

            states = {'state_dict': global_model.state_dict()}
            torch.save(states, checkpoint_path)
            states_head = {'state_dict': model_head.state_dict()}
            torch.save(states_head, head_checkpoint_path)

            print(f'\n---------------------Test inference after {r+1} rounds of training -----------------------')
            for num, idx in enumerate(idxs_users):
                # for each local device
                performance_dict = dict()
                performance_dict['round'] = r+1
                performance_dict['user'] = idx
                performance_dict['training_loss'] = loss_avg
                score, th = test_inference(args, global_model,
                                           BaIoT_data(x_train_n_ls[idx], np.zeros(len(x_train_n_ls[idx]), dtype=int)),
                                           BaIoT_data(x_val_n_ls[idx], np.zeros(len(x_val_n_ls[idx]), dtype=int)),
                                           BaIoT_data(x_test_ls[idx], y_test_ls[idx])
                                           )

                split_evaluate(y_test_ls[idx][0:len(score)], score,
                               filename=f'{args.result_folder}contrastive_r{r}' + save_name,
                               manual_th=th,
                               perform_dict=performance_dict)

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
        model = generate_model(args, input_size=x_train_normal.shape[1])
        resume_path = os.path.join(args.checkpoint_folder, f'best_model_{args.model_type}.pth')
        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])
        percents = [0.4, 0.4, 0.3, 0.61, 0.57 ]

        for idx in range(args.num_users):
            # test the accuracy of global model on the local data at each device

            score, th = test_inference(args, model,
                                       BaIoT_data(x_train_n_ls[idx], np.zeros(len(x_train_n_ls[idx]), dtype=int)),
                                       BaIoT_data(x_val_n_ls[idx], np.zeros(len(x_val_n_ls[idx]), dtype=int)),
                                       BaIoT_data(x_test_ls[idx], y_test_ls[idx]),
                                       percent=percents[idx])
            print('the IoT device is', device_names[idx])
            split_evaluate(y_test_ls[idx][0:len(score)], score, plot=True,
                           filename=f'{args.result_folder}contrastive_final',
                           manual_th=th)


if __name__ == "__main__":
    DEVICE_NAMES = ['Danmini_Doorbell', 'Ecobee_Thermostat',
                    'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
                    'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
                    'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
                    'SimpleHome_XCS7_1003_WHT_Security_Camera']

    gv.init('fl')
    args = gv.args
    args.manual_seed = 1
    # device_names = ['Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera']
    device_names = ['SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']
    # device_names = ['Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
    #                 'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
    #                 'SimpleHome_XCS7_1003_WHT_Security_Camera'
    #                 ]

    args.data_distribution = 'iid'
    args.learning_rate = 0.001
    args.num_users = len(device_names)
    args.local_epochs = 4
    args.epochs = 12
    args.frac = 1
    args.lr_decay = 80

    args.tau = 0.1
    args.n_train_batch_size = 8

    # args.mode = 'train'
    # main(args, device_names)

    args.mode = 'test'
    main(args, device_names)


