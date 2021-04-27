import torch
import torch.backends.cudnn as cudnn

import os
import numpy as np
import argparse

from test import get_normal_vector, split_acc_diff_threshold, cal_score, cal_score_downstream,\
    get_new_represent
from utils.utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize,\
    post_process, evaluate, get_score, get_threshold
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.setup_NSL import NSL_KDD, NSL_data
from model import generate_model
from models import mlp
# from models import resnet, shufflenet, shufflenetv2, mobilenet, mobilenetv2
import ast
from utils.utils import split_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='DAD training on Videos')

    parser.add_argument('--mode', default='test', type=str, help='train | test(validation)')
    parser.add_argument('--root_path', default='', type=str, help='root path of the dataset')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--latent_dim', default=64, type=int, help='contrastive learning dimension')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')

    parser.add_argument('--model_type', default='mlp', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')

    parser.add_argument('--epochs', default=20, type=int, help='Number of total epochs to run')
    parser.add_argument('--val_step', default=4, type=int, help='validate per val_step epochs')
    parser.add_argument('--save_step', default=4, type=int, help='checkpoint will be saved every save_step epochs')
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

    parser.add_argument('--n_threads', default=1, type=int, help='num of workers loading dataset')
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
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval, help='True|False: a flag controlling whether to create a new log file')
    parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    parser.add_argument('--score_folder', default='./result/score/', type=str, help='folder to store scores')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
    parser.add_argument('--width_mult', default=2.0, type=float,
                        help='hyper-parameters when using shufflenet|mobilenet')

    parser.add_argument('--n_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--a_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')

    args = parser.parse_args()
    return args


def train(train_normal_loader, train_anormal_loader, model, model_head, nce_average, criterion, optimizer, epoch, args,
          batch_logger, epoch_logger, memory_bank=None):
    losses = AverageMeter()
    prob_meter = AverageMeter()

    model.train()
    model_head.train()
    for batch, ((normal_data, idx_n), (anormal_data, idx_a)) in enumerate(
            zip(train_normal_loader, train_anormal_loader)):
        if normal_data.size(0) != args.n_train_batch_size:
            break
        data = torch.cat((normal_data, anormal_data), dim=0)  # n_vec as well as a_vec are all normalized value
        if args.use_cuda:
            data = data.cuda()
            idx_a = idx_a.cuda()
            idx_n = idx_n.cuda()
            normal_data = normal_data.cuda()

        # ================forward====================
        unnormed_vec, normed_vec = model(data.float())
        vec = model_head(unnormed_vec)
        n_vec = vec[0:args.n_train_batch_size]
        a_vec = vec[args.n_train_batch_size:]
        outs, probs = nce_average(n_vec, a_vec, idx_n, idx_a, normed_vec[0:args.n_train_batch_size])
        loss = criterion(outs)

        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===========update memory bank===============
        model.eval()
        _, n = model(normal_data.float())
        n = n.detach()
        average = torch.mean(n, dim=0, keepdim=True)
        if len(memory_bank) < args.memory_bank_size:
            memory_bank.append(average)
        else:
            memory_bank.pop(0)
            memory_bank.append(average)
        model.train()

        # ===============update meters ===============
        losses.update(loss.item(), outs.size(0))
        prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'probs': prob_meter.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        if batch % 10 == 0:
            print(f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} '
                  f'| Loss: {losses.val} ({losses.avg})| Probs: {prob_meter.val} ({prob_meter.avg})')
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'probs': prob_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    return memory_bank, losses.avg


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.normvec_folder):
        os.makedirs(args.normvec_folder)
    if not os.path.exists(args.score_folder):
        os.makedirs(args.score_folder)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    anormal_data = NSL_KDD([('DoS', 0.0)], data_type='anomaly')
    normal_data = NSL_KDD([('DoS', 0.0)], data_type='normal')
    all_data = NSL_KDD([('DoS', 0.0)], data_type=None)

    if args.mode == 'train':
        print("=================================Loading Anormaly Training Data!=================================")
        training_anormal_data = NSL_data(anormal_data.train_data, anormal_data.train_labels)
        training_anormal_size = int(len(training_anormal_data) * args.a_split_ratio)
        training_anormal_data = torch.utils.data.Subset(training_anormal_data, np.arange(training_anormal_size))
        a = anormal_data.train_data
        b = all_data.train_data
        a = training_anormal_data[1]

        train_anormal_loader = torch.utils.data.DataLoader(
            training_anormal_data,
            batch_size=args.a_train_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("=================================Loading Normal Training Data!=================================")
        training_normal_data = NSL_data(normal_data.train_data, normal_data.train_labels)
        training_normal_size = int(len(training_normal_data) * args.n_split_ratio)
        training_normal_data = torch.utils.data.Subset(training_normal_data, np.arange(training_normal_size))

        train_normal_loader = torch.utils.data.DataLoader(
            training_normal_data,
            batch_size=args.n_train_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("========================================Loading Validation Data========================================")

        validation_data = NSL_data(all_data.validation_data, all_data.validation_labels)
        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        len_neg = training_anormal_data.__len__()
        len_pos = training_normal_data.__len__()
        num_val_data = validation_data.__len__()
        print(f'len_neg: {len_neg}')
        print(f'len_pos: {len_pos}')

        print("=================================== Loading Test Data =====================================")
        test_data = NSL_data(all_data.test_data, all_data.test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("=====================================Generating Model=========================================")

        model_head = mlp.ProjectionHead(args.latent_dim, args.feature_dim)

        if args.use_cuda:
            model_head.cuda()

        if args.resume_path == '':
            # ===============generate new model or pre-trained model===============
            model = generate_model(args)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                        dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
            nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
            criterion = NCECriterion(len_neg)
            begin_epoch = 1
            best_acc = 0
            memory_bank = []
        else:
            # ===============load previously trained model ===============
            args.pre_train_model = False
            model = generate_model(args)
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

        print("====================================!!!START TRAINING!!!====================================")
        cudnn.benchmark = True
        batch_logger = Logger(os.path.join(args.log_folder, 'batch.log'), ['epoch', 'batch', 'loss', 'probs', 'lr'],
                              args.log_resume)
        epoch_logger = Logger(os.path.join(args.log_folder, 'epoch.log'), ['epoch', 'loss', 'probs', 'lr'],
                              args.log_resume)
        val_logger = Logger(os.path.join(args.log_folder, 'val.log'),
                            ['epoch', 'accuracy', 'normal_acc', 'anormal_acc', 'threshold', 'acc_list',
                             'normal_acc_list', 'anormal_acc_list'], args.log_resume)

        for epoch in range(begin_epoch, begin_epoch + args.epochs + 1):
            memory_bank, loss = train(train_normal_loader, train_anormal_loader, model, model_head, nce_average,
                                      criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)

            if epoch % args.val_step == 0:

                print("=============================!!!Evaluating!!!===================================")
                normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
                normal_vec = l2_normalize(normal_vec)

                model.eval()

                accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                    model, normal_vec, test_loader, args.use_cuda)
                print(f'testing Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n}'
                      f' | Anormal Acc: {acc_a} | Threshold: {best_threshold}')

                accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                    model, normal_vec, validation_loader, args.use_cuda)
                print(f'Validation Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n}'
                      f' | Anormal Acc: {acc_a} | Threshold: {best_threshold}')

                print("===============================!!!Logging!!!==================================")
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
                    print("======================================!!!Saving!!!====================================")
                    checkpoint_path = os.path.join(args.checkpoint_folder,
                                                   f'best_model_{args.model_type}.pth')
                    states = {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'acc': accuracy,
                        'threshold': best_threshold,
                        'nce_average': nce_average,
                        'memory_bank': memory_bank
                    }
                    torch.save(states, checkpoint_path)

                    head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                        f'best_model_{args.model_type}_head.pth')
                    states_head = {'state_dict': model_head.state_dict()}
                    torch.save(states_head, head_checkpoint_path)

            if epoch % args.save_step == 0:
                print("=====================================!!!Saving!!!=======================================")
                checkpoint_path = os.path.join(args.checkpoint_folder,
                                               f'{args.model_type}_{epoch}.pth')
                states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': accuracy,
                    'nce_average': nce_average,
                    'memory_bank': memory_bank
                }
                torch.save(states, checkpoint_path)

                head_checkpoint_path = os.path.join(args.checkpoint_folder,
                                                    f'{args.model_type}_{epoch}_head.pth')
                states_head = {'state_dict': model_head.state_dict()}
                torch.save(states_head, head_checkpoint_path)

            if epoch % args.lr_decay == 0:
                lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay))
                adjust_learning_rate(optimizer, lr)
                print(f'New learning rate: {lr}')

    elif args.mode == 'test':
        if not os.path.exists(args.normvec_folder):
            os.makedirs(args.normvec_folder)
        score_folder = args.score_folder
        args.pre_train_model = False

        model = generate_model(args)
        resume_path = './checkpoints/best_model_' + args.model_type + '.pth'
        # model_id = 9
        # resume_path = './checkpoints/' + args.model_type + '_{}.pth'.format(model_id*10)
        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])

        model.eval()

        print("================================ Loading Normal Data =====================================")
        training_normal_data = NSL_data(normal_data.train_data, normal_data.train_labels)
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

        validation_data = NSL_data(normal_data.validation_data, normal_data.validation_labels)
        validation_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("================================ Loading Test Data =====================================")
        test_data = NSL_data(all_data.test_data, all_data.test_labels)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        print("=================================START EVALUATING=========================================")
        normal_vec = get_normal_vector(model, train_normal_loader_for_test,
                                       args.cal_vec_batch_size,
                                       args.latent_dim,
                                       args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec.npy'), normal_vec.cpu().numpy())

        # compute a decision-making threshold using the validation dataset
        valid_scores = cal_score(model, normal_vec, validation_loader, None, args.use_cuda)
        th = get_threshold(valid_scores, percent=3)
        print(f'the threshold is set as {th}')

        # evaluating the scores of the test dataset and show the IDS performance
        cal_score(model, normal_vec, test_loader, score_folder, args.use_cuda)
        score = get_score(score_folder)
        best_acc, best_threshold, AUC = evaluate(score, all_data.test_labels, False, model_id='best')
        print(f'Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)}')
        split_evaluate(all_data.test_labels, score, plot=True, filename='./result/detection/contrastive', manual_th=th)


