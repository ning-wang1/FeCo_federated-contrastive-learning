import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models import mlp
import global_vars as gv
import json
import csv
import os
import time
import random
from utils.setup_NSL import NSL_KDD, NSL_data
import numpy as np


def save_statistics(experiment_name, line_to_add, filename="summary_statistics.csv", create=False):
    summary_filename = "{}/{}".format(experiment_name, filename)
    if create:
        with open(summary_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)
    else:
        with open(summary_filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line_to_add)

    return summary_filename


class MLPDW(nn.Module):
    def __init__(self, args, pretrain_path, input_dim=110, output_dim=2):
        super(MLPDW, self).__init__()

        self.base = mlp.get_model(input_size=input_dim, layer1_size=128, layer2_size=256, output_size=args.latent_dim)

        resume_checkpoint = self.load_compat(pretrain_path)
        self.base.load_state_dict(resume_checkpoint)
        for p in self.parameters():
            p.requires_grad = False

        self.hidden = nn.Linear(args.latent_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x, _ = self.base(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)

        return x

    def load_compat(self, model_file_path):
        """
        load parameters of pre_train model
        :param model_file_path: file path of the pre_train_model.
        :param remove_fc_layer: a flag indicating whether to remove the fully connected layer
        """
        state = torch.load(model_file_path, map_location='cuda:0')
        param_dict = state['state_dict']

        param_dict_new = dict()
        # param_dict = state['module']

        for p in param_dict.keys():
            p_new = p.split(".", 1)[1]
            param_dict_new[p_new] = param_dict[p]

        return param_dict_new


class ExperimentBuilder(object):
    def __init__(self, device, args, model):
        self.device = device
        self.use_cuda = args.use_cuda
        self.args = args
        self.model = model
        self.model.to(self.device)

        self.best_acc = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=args.epochs,
                                                              eta_min=.001)
        # self.experiment_path, self.logs_fp = self.build_log_path()
        self.train_sum_name = 'train_summary.csv'
        self.test_sum_name = 'test_summary.csv'

        if torch.cuda.is_available():
            cudnn.benchmark = True

    def build_log_path(self):
        """
        build log file direction and return the log filepath
        """
        experiment_path = os.path.abspath(self.args.experiment_name)
        logs_fp = "{}/{}".format(experiment_path, "pure_train")
        if not os.path.exists(logs_fp):
            os.makedirs(logs_fp)
        return experiment_path, logs_fp

    def new_log_file(self):
        """
        build attributes of log file for training and testing
        """
        train_stat_names = ['epoch', 'avg_train_loss', 'train_acc_mean', 'train_correct', 'train_total',
                            'avg_val_loss', 'val_acc_mean', 'val_correct', 'val_total']
        test_stat_names = ['avg_test_loss', 'test_acc_mean', 'test_correct', 'test_total']

        train_fp = os.path.join(self.logs_fp, self.train_sum_name)
        test_fp = os.path.join(self.logs_fp, self.test_sum_name)

        if not (os.path.exists(train_fp) and os.path.exists(test_fp)):
            save_statistics(self.logs_fp, train_stat_names, filename=self.train_sum_name, create=True)
            save_statistics(self.logs_fp, test_stat_names, filename=self.test_sum_name, create=True)

    def get_grad_layer_num(self):
        # return the number of layers that have grad
        i = 0
        for tensor_name, tensor in self.model.named_parameters():
            if 'pool' not in tensor_name:
                i += 1
        return i

    def train(self, epoch, train_loader):
        """
        Model Training
        """
        print('Epoch {}/{}'.format(epoch + 1, self.args.epochs))
        print('-' * 10)
        start_time = time.time()
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs.float())
            loss = F.cross_entropy(input=outputs, target=targets.long())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 100 == 0:
                print('Batch {}/{}: TrainLoss: {}'.format(batch_idx, len(train_loader), train_loss/(batch_idx+1)))
        self.scheduler.step()
        avg_loss = train_loss / (batch_idx + 1)
        acc = 100. * correct / total
        end_time = time.time()

        stat = [epoch, avg_loss, acc, correct, total]
        #save_statistics(self.logs_fp, [avg_loss, acc, correct, total], filename="train_summary.csv")
        print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (
            avg_loss, acc, correct, total, end_time - start_time))
        return stat

    def test(self, model, val_loader):
        """
        Model Testing or evaluation
        param: model: if model is None evaluate the current model, else evaluate the input model
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        if model is None:
            test_model = self.model
        else:
            test_model = model

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = test_model(inputs.float())
                loss = F.cross_entropy(outputs, targets.long())
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        avg_loss = test_loss / (batch_idx + 1)
        stat = [avg_loss, acc, correct, total]
        if model is None:
            print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (avg_loss, acc, correct, total))
            # Save checkpoint.
            if acc > self.best_acc:
                print('Saving..')
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(self.model.state_dict(), './checkpoint/ckpt.pth')
                self.best_acc = acc
        else:
            save_statistics(self.logs_fp, stat, filename=self.test_sum_name)
            print('\nThe Best Result>>>>>>>TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (avg_loss, acc, correct, total))
        return stat

    def run_experiment(self, train_loader, test_loader):
        """
        organize the experiment run, validation and test
        """

        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):

            train_stat = self.train(epoch, train_loader=train_loader)
            val_stat = self.test(model=None, val_loader=test_loader)
            train_stat.extend(val_stat)
            # save_statistics(self.logs_fp, train_stat, filename=self.train_sum_name)

        # Loading weight files to the model and testing them.
        net_test = self.model
        net_test = net_test.to(self.device)
        net_test.load_state_dict(torch.load('./checkpoint/ckpt.pth'))

        test_stat = self.test(model=net_test, val_loader=test_loader)
        acc = test_stat[1]
        return acc


if __name__ == '__main__':
    gv.init('centralized')
    args = gv.args

    all_data = NSL_KDD(data_type=None)

    x_train = all_data.train_data
    x_test = all_data.test_data
    y_train = all_data.y_train_multi_class
    y_test = all_data.y_test_multi_class
    # y_train = all_data.train_labels
    # y_test = all_data.test_labels

    test_data = NSL_data(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=64,
        shuffle=False)

    train_data = NSL_data(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=64,
        shuffle=True)

    model = MLPDW(
        args,
        pretrain_path=os.path.join(args.checkpoint_folder, f'best_model_{args.model_type}.pth'),
        input_dim=all_data.train_data.shape[1],
        output_dim=5)

    exp = ExperimentBuilder('cuda: 0', args, model)

    exp.run_experiment(train_loader, test_loader)
