import torch
import torch.backends.cudnn as cudnn

import os
import csv
import numpy as np
import global_vars as gv

from test import get_normal_vector, split_acc_diff_threshold, cal_score, cal_score_downstream,\
    get_new_represent, train_downstream_classifier, multiclass_test, detect_with_manifold
from utils.utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize,\
    post_process, evaluate, get_score, get_threshold
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.setup_NSL import NSL_KDD, NSL_data
from model import generate_model
from models import mlp
# from models import resnet, shufflenet, shufflenetv2, mobilenet, mobilenetv2
from utils.utils import split_evaluate, per_class_acc, check_recall, set_random_seed, split_evaluate_two_steps


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


def main(args):
    rng = set_random_seed(args.manual_seed, args.use_cuda)
    if args.nesterov:
        dampening = 1
    else:
        dampening = args.dampening

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

    if args.mode == 'train':
        print("=================================Loading Anormaly Training Data!=================================")
        training_anormal_data = NSL_data(anormal_data.train_data, anormal_data.train_labels)
        training_anormal_size = int(len(training_anormal_data) * args.a_split_ratio)
        training_anormal_data = torch.utils.data.Subset(training_anormal_data, np.arange(training_anormal_size))

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
            model = generate_model(args, input_size=all_data.train_data.shape[1])
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
            model = generate_model(args, input_size=all_data.train_data.shape[1])
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

        for epoch in range(begin_epoch, begin_epoch + args.epochs):
            memory_bank, loss = train(train_normal_loader, train_anormal_loader, model, model_head, nce_average,
                                      criterion, optimizer, epoch, args, batch_logger, epoch_logger, memory_bank)

            if epoch % args.val_step == 0:

                print("=============================!!!Evaluating!!!===================================")
                normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
                normal_vec = l2_normalize(normal_vec)

                model.eval()

                # accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
                #     model, normal_vec, test_loader, args.use_cuda)
                # print(f'testing Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} | Normal Acc: {acc_n}'
                #       f' | Anormal Acc: {acc_a} | Threshold: {best_threshold}')

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
                # if accuracy > best_acc:
                if True:
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
                        'nce_average': nce_average.state_dict(),
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
                    'nce_average': nce_average.state_dict(),
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

        args.pre_train_model = False

        model = generate_model(args, input_size=all_data.train_data.shape[1])
        resume_path = os.path.join(args.checkpoint_folder,
                                   f'best_model_{args.model_type}.pth')
        # model_id = 9
        # resume_path = './checkpoints/' + args.model_type + '_{}.pth'.format(model_id*10)
        resume_checkpoint = torch.load(resume_path)
        model.load_state_dict(resume_checkpoint['state_dict'])

        model.eval()

        # multiclass_test(args, all_data, model)

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

        training_anormal_data = NSL_data(anormal_data.train_data, anormal_data.train_labels)
        train_anormal_loader_for_test = torch.utils.data.DataLoader(
            training_anormal_data,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )

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

        # normal_len = len(training_normal_data)
        # anormal_len = len(training_anormal_data)
        # test_len = len(test_data)
        # pred_consist = detect_with_manifold(args, model, train_normal_loader_for_test,
        #                                     train_anormal_loader_for_test, test_loader,
        #                                     normal_len, anormal_len, test_len, all_data.test_labels)

        print("=================================START EVALUATING=========================================")
        normal_vec = get_normal_vector(model, train_normal_loader_for_test,
                                       args.cal_vec_batch_size,
                                       args.latent_dim,
                                       args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec.npy'), normal_vec.cpu().numpy())

        # compute a decision-making threshold using the validation dataset
        valid_scores = cal_score(model, normal_vec, validation_loader, None, args.use_cuda)
        th = get_threshold(valid_scores, percent=5)
        print(f'the threshold is set as {th}')

        # evaluating the scores of the test dataset and show the IDS performance
        score_folder = args.score_folder
        cal_score(model, normal_vec, test_loader, score_folder, args.use_cuda)
        score = get_score(score_folder)

        # split_evaluate_two_steps(pred_consist, all_data.test_labels, score, manual_th=th, perform_dict=None)

        performance_dict = dict()
        split_evaluate(all_data.test_labels, score, plot=True,
                       filename=f'{args.result_folder}contrastive', manual_th=th, perform_dict=performance_dict)
        per_class_acc(all_data.y_test_multi_class, score, th, perform_dict=performance_dict)

        with open(f'{args.score_folder}metrics{args.manual_seed}.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=performance_dict.keys())
            writer.writeheader()
            writer.writerow(performance_dict)

        # train_downstream_classifier(args, model, all_data.train_data, all_data.y_train_multi_class,
        #                             all_data.test_data, all_data.y_test_multi_class)
        # train_downstream_classifier(args, model, all_data.train_data, all_data.train_labels,
        #                             all_data.test_data, all_data.test_labels)


if __name__ == '__main__':
    gv.init('centralized')
    args = gv.args
    args.manual_seed = 2

    args.data_partition_type = 'normalOverAll'

    if args.data_partition_type is 'normalOverAll':
        # args.epochs = 60
        # args.val_step = 30
        # args.save_step = 30
        # args.tau = 0.03
        # args.learning_rate = 0.001
        # args.lr_decay = 30
        # args.n_train_batch_size = 6

        args.epochs = 60
        args.val_step = 30
        args.save_step = 30
        args.tau = 0.03
        args.learning_rate = 0.001
        args.lr_decay = 45
        args.n_train_batch_size = 5

    else:
        args.epochs = 50
        args.val_step = 10
        args.save_step = 10
    #
    args.mode = 'train'
    main(args)
    args.mode = 'test'
    main(args)

