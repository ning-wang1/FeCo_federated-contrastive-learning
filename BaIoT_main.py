import os
import csv
import numpy as np
import global_vars as gv
import torch
import torch.backends.cudnn as cudnn


from models import mlp
from model import generate_model
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from centralized_main import train
from utils.setup_BaIoT import BaIoT, BaIoT_data
from utils.utils import split_evaluate, adjust_learning_rate,\
    Logger, l2_normalize, get_score, get_threshold
from test import get_normal_vector, split_acc_diff_threshold, cal_score, save_score_with_label

DEVICE_NAMES = ['Danmini_Doorbell', 'Ecobee_Thermostat',
               'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
               'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
               'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
               'SimpleHome_XCS7_1003_WHT_Security_Camera']

PERCENTAGE = {'Danmini_Doorbell': 0.1, 'Ecobee_Thermostat': 0.1,
               'Ennio_Doorbell': 0.5, 'Philips_B120N10_Baby_Monitor': 0.3,
               'Provision_PT_737E_Security_Camera': 0.3, 'Provision_PT_838_Security_Camera': 0.3,
               'Samsung_SNH_1011_N_Webcam': 0.3, 'SimpleHome_XCS7_1002_WHT_Security_Camera': 0.3,
               'SimpleHome_XCS7_1003_WHT_Security_Camera': 0.3}


def main(args, device_name, test_model=None):

    if args.nesterov:
        dampening = 1
    else:
        dampening = args.dampening

    data = BaIoT(device_name)
    x_train_attack, x_train_normal = data.x_train_attack, data.x_train_normal
    x_val_attack, x_val_normal = data.x_val_attack, data.x_val_normal
    x_val, y_val, x_test, y_test = data.x_val, data.y_val, data.x_test, data.y_test

    y_train_attack = np.ones(x_train_attack.shape[0], dtype=int)
    y_train_normal = np.zeros(x_train_normal.shape[0], dtype=int)
    y_val_normal = np.zeros(x_val_normal.shape[0], dtype=int)

    if args.mode == 'train':
        print("==================== Loading Anormaly Training Data =====================")
        training_anormal_data = BaIoT_data(x_train_attack, y_train_attack)
        training_anormal_size = int(len(training_anormal_data) * args.a_split_ratio)
        training_anormal_data = torch.utils.data.Subset(training_anormal_data, np.arange(training_anormal_size))

        train_anormal_loader = torch.utils.data.DataLoader(
            training_anormal_data,
            batch_size=args.a_train_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("======================= Loading Normal Training Data ===================")
        training_normal_data = BaIoT_data(x_train_normal, y_train_normal)
        training_normal_size = int(len(training_normal_data) * args.n_split_ratio)
        training_normal_data = torch.utils.data.Subset(training_normal_data, np.arange(training_normal_size))

        train_normal_loader = torch.utils.data.DataLoader(
            training_normal_data,
            batch_size=args.n_train_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("===================== Loading validation Data =======================")
        validation_data = BaIoT_data(x_val, y_val)
        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("======================= Generating Model ==========================")
        model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)
        len_neg = training_anormal_data.__len__()
        len_pos = training_normal_data.__len__()
        print(f'len_neg: {len_neg}')
        print(f'len_pos: {len_pos}')

        if args.use_cuda:
            model_head.cuda()

        if args.resume_path == '':
            # generate new model or pre-trained model
            model = generate_model(args, input_size=x_train_normal.shape[1])
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                        dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
            nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
            criterion = NCECriterion(len_neg)
            begin_epoch = 1
            best_acc = 0
            memory_bank = []
        else:
            # load previously trained model
            args.pre_train_model = False
            model = generate_model(args, input_size=x_train_normal.shape[1])
            resume_path = os.path.join(args.checkpoint_folder, args.resume_path)
            resume_checkpoint = torch.load(resume_path)
            model.load_state_dict(resume_checkpoint['state_dict'])
            resume_head_checkpoint = torch.load(os.path.join(args.checkpoint_folder, args.resume_head_path))
            model_head.load_state_dict(resume_head_checkpoint['state_dict'])
            if args.use_cuda:
                model_head.cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                        dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
            optimizer.load_state_dict(resume_checkpoint['optimizer'])
            nce_average = resume_checkpoint['nce_average']
            criterion = NCECriterion(len_neg)
            begin_epoch = resume_checkpoint['epoch'] + 1
            best_acc = resume_checkpoint['acc']
            memory_bank = resume_checkpoint['memory_bank']
            del resume_checkpoint
            torch.cuda.empty_cache()
            adjust_learning_rate(optimizer, args.learning_rate)

        print("========================= START TRAINING ===========================")
        cudnn.benchmark = True
        batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                              args.log_resume)
        epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                              args.log_resume)
        val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                            ['epoch', 'accuracy', 'normal_acc', 'anormal_acc', 'threshold', 'acc_list',
                             'normal_acc_list', 'anormal_acc_list'], args.log_resume)

        for epoch in range(begin_epoch, begin_epoch + args.epochs):
            memory_bank, loss = train(train_normal_loader, train_anormal_loader, model, model_head, nce_average,
                                      criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)

            if epoch % args.val_step == 0:

                print("======================== Evaluating ===========================")
                normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
                normal_vec = l2_normalize(normal_vec)

                model.eval()

                accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                    model, normal_vec, validation_loader, args.use_cuda)
                print(f'Validation Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n}'
                      f' | Anormal Acc: {acc_a} | Threshold: {best_threshold}')

                print("========================= Logging ===========================")
                val_logger.log({
                    'epoch': epoch,
                    'accuracy': accuracy * 100,
                    'normal_acc': acc_n * 100,
                    'anormal_acc': acc_a * 100,
                    'threshold': best_threshold,
                    'acc_list': acc_list,
                    'normal_acc_list': acc_n_list,
                    'anormal_acc_list': acc_a_list
                })
                if accuracy > best_acc:
                    best_acc = accuracy
                    print("=========================== Saving ==========================")
                    checkpoint_path = os.path.join(args.checkpoint_folder,
                                                   f'best_model_{device_name}.pth')
                    states = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'acc': accuracy,
                        'threshold': best_threshold,
                        'nce_average': nce_average.state_dict(),
                        'memory_bank': memory_bank
                     }
                    torch.save(states, checkpoint_path)

                    head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                        f'best_model_{device_name}_head.pth')
                    states_head = {'state_dict': model_head.state_dict()}
                    torch.save(states_head, head_checkpoint_path)

            if epoch % args.lr_decay == 0:
                lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay))
                adjust_learning_rate(optimizer, lr)
                print(f'New learning rate: {lr}')

    elif args.mode == 'test':
        if not os.path.exists(args.normvec_folder):
            os.makedirs(args.normvec_folder)

        args.pre_train_model = False

        model = generate_model(args, input_size=x_train_normal.shape[1])
        if test_model is not None:
            resume_path = os.path.join(args.checkpoint_folder,
                                       f'best_model_{test_model}.pth')
        else:
            resume_path = os.path.join(args.checkpoint_folder,
                                       f'best_model_{device_name}.pth')

        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])

        # Export the model
        # dummy = torch.autograd.Variable(torch.randn(1, 115, device='cuda'))
        #
        # torch.onnx.export(model.module,  # model being run
        #                   dummy,  # model input (or a tuple for multiple inputs)
        #                   "checkpoints/model1.onnx",  # where to save the model (can be a file or file-like object)
        #                   export_params=True,  # store the trained parameter weights inside the model file
        #                   opset_version=10,  # the ONNX version to export the model to
        #                   do_constant_folding=True,  # whether to execute constant folding for optimization
        #                   input_names=['input'],  # the model's input names
        #                   output_names=['output'],  # the model's output names
        #                   dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
        #                                 'output': {0: 'batch_size'}})

        model.eval()

        print("====================== Loading Normal Data ===========================")
        training_normal_data = BaIoT_data(x_train_normal, y_train_normal)
        training_normal_size = int(len(training_normal_data) * args.n_split_ratio)
        training_normal_data = torch.utils.data.Subset(training_normal_data, np.arange(training_normal_size))

        train_normal_loader_for_test = torch.utils.data.DataLoader(
            training_normal_data,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        val_normal_data = BaIoT_data(x_val_normal, y_val_normal)
        val_normal_loader = torch.utils.data.DataLoader(
            val_normal_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
            drop_last=True,
        )

        print("======================== Loading Test Data ============================")
        test_data = BaIoT_data(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
            drop_last=True,
        )

        # compute a decision-making threshold using the validation dataset
        print("=========================== START EVALUATING ==========================")
        normal_vec = get_normal_vector(model, train_normal_loader_for_test,
                                       args.cal_vec_batch_size,
                                       args.latent_dim,
                                       args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec.npy'), normal_vec.cpu().numpy())
        valid_scores = cal_score(model, normal_vec, val_normal_loader, None, args.use_cuda)
        th = get_threshold(valid_scores, percent=0.32)
        print(f'the threshold is set as {th}')

        with open('checkpoints/vec1.npy', 'wb') as f:
            np.save(f, normal_vec.cpu().numpy())
        with open('checkpoints/threshold1.npy', 'wb') as f:
            np.save(f, [th])

        # evaluating the scores of the test dataset and show the IDS performance
        score_folder = args.score_folder
        score = cal_score(model, normal_vec, test_loader, os.path.join(score_folder, 'score.npy'), args.use_cuda)
        save_path = os.path.join(score_folder,  f'score_label_{device_name}.npy')
        save_score_with_label(score, y_test[:len(score)], save_path)

        performance_dict = dict()
        split_evaluate(y_test[:len(score)], score, plot=True,
                       filename=f'{args.result_folder}contrastive',
                       manual_th=th, perform_dict=performance_dict)

        with open(f'{args.score_folder}metrics{device_name}_{args.manual_seed}.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
            writer.writeheader()
            writer.writerow(performance_dict)


if __name__ == '__main__':
    gv.init('centralized')
    args = gv.args
    seed = 0
    args.manual_seed = seed
    np.random.seed(seed)

    args.epochs = 30
    args.val_step = 30
    args.save_step = 30
    args.tau = 0.1
    args.learning_rate = 0.001
    args.lr_decay = 45
    args.n_train_batch_size = 8

    # device_name = 'Danmini_Doorbell'
    device_name = DEVICE_NAMES[1]
    print(f'Device name: {device_name}')
    # args.mode = 'train'
    # main(args, device_name)
    args.mode = 'test'
    main(args, device_name)
    # main(args, device_name, test_model=DEVICE_NAMES[1])

