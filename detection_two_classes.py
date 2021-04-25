# train_models.py -- train the ids models
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os.path
import numpy as np
import logging
import random
import pickle
import argparse

from utils.setup_NSL import NSL_KDD, NSL_data
from models.mlp import MLP
from utils import classifier as clf
from utils.classifier import get_score
from utils.utils import split_evaluate

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Log setting
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", level=logging.INFO)


def get_acc(mlp, loader):
    correct = 0
    for i, data in enumerate(loader, 0):
        input, target = data
        input = Variable(input)
        output, _ = mlp(input.float())
        pred = torch.max(output, 1)[1].data.numpy()
        correct += float((pred == target.data.numpy()).astype(int).sum())
    total = len(loader.dataset)
    return correct / total


def mlp_predict(mlp, loader):

    for i, data in enumerate(loader, 0):
        input, target = data
        input = Variable(input)
        output, _ = mlp(input.float())
        pred = output.data.numpy()
        if i == 0:
            y_pred = pred
        else:
            y_pred = np.concatenate([y_pred, pred])

    return y_pred


def early_stop(lst, ma):
    length = len(lst)
    sum_a = sum(lst[length - ma:length])
    length = length - 1
    sum_b = sum(lst[length - ma:length])

    if sum_a < sum_b:
        return True
    else:
        return False


def train_ids(input_shape, trainloader, validationloader, testloader, save_folder):
    """
    Standard neural network training procedure.
    """
    max_epochs = 10
    moving_average = 4

    mlp = MLP(input_shape, 128, 256)
    print(mlp)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.0001, weight_decay=0.001)

    mlp_list = []
    crt_list = []

    for epoch in range(0, max_epochs):

        current_loss = 0
        for i, data in enumerate(trainloader, 0):
            input, target = data
            input, target = Variable(input), Variable(target)

            mlp.zero_grad()
            output, _ = mlp(input.float())
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            current_loss += loss

        print('[ %d ]Training loss : %.3f' % (epoch+1, current_loss))
        train_c = get_acc(mlp, trainloader)
        test_c = get_acc(mlp, testloader)
        valid_c = get_acc(mlp, validationloader)
        print('[ %d ] Training accuracy: %.4f, Validation Acc%.4f, Testing Acc: %.4f' %
              (epoch + 1, train_c, valid_c, test_c))

        mlp_list.append(mlp)
        crt_list.append(valid_c)

        if epoch >= moving_average:
            if early_stop(crt_list, moving_average):
                print('Early stopping.')
                index = int(len(mlp_list) - moving_average / 2)
                torch.save(mlp.state_dict(), os.path.join(save_folder, 'nn'))
                return mlp_list[index]

        current_loss = 0

    return mlp


def present_result(test_data, test_labels, model_dict):

    performance_dict = {}
    sample_num = test_labels.shape[0]
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        if model_name is 'Consistency':
            clf.evaluate_sub(model_name, test_labels, model.predict(test_data))
            pred = model.predict(test_data)
        elif model_name is 'NN':
            model.evaluate_only(test_data, test_labels)
            pred = model.predict(test_data).eval()
            pred = np.argmax(pred, axis=1)
        else:
            clf.evaluate_only(model_name, model, test_data, test_labels)
            pred = model.predict(test_data)
        pred =pred.reshape([-1, 1])
        correct_num = np.sum(pred==test_labels)
        misclassified_num = test_labels.shape[0] - correct_num
        performance_dict[model_name] = [correct_num, misclassified_num, sample_num]

    return performance_dict


def present_split_acc(y_pred, y):
    if -1 in y:
        y = ((y + 1) / 2).astype(int)

    total_n = np.sum(y)
    total_a = len(y) - total_n

    correct = y_pred == y
    correct_a = np.sum(correct[np.where(data.test_labels == 0)])
    correct_n = np.sum(correct[np.where(data.test_labels == 1)])

    acc_n = correct_n / total_n
    acc_a = correct_a / total_a
    acc = (correct_n + correct_a) / (total_n + total_a)

    print('Split Acc under the Best Acc: normal_acc={:.3f}, anormal_acc={:.3f}'.format(acc_n, acc_a))
    print('overall acc: {:.3f}'.format(acc))


def train_discriminators(train_data, train_labels, save_folder):

    lgr_model = clf.classifier('LGR', train_data, train_labels)
    knn_model = clf.classifier('KNN', train_data, train_labels)
    bnb_model = clf.classifier('BNB', train_data, train_labels)
    svm_model = clf.classifier('SVM', train_data, train_labels)
    dtc_model = clf.classifier('DTC', train_data, train_labels)
    mlp_model = clf.classifier('MLP', train_data, train_labels)

    models = {'LGR': lgr_model, 'KNN': knn_model, 'BNB': bnb_model, 'SVM': svm_model, 'DTC': dtc_model,
              'MLP': mlp_model}
    for model_name, model in models.items():
        pickle.dump(model, open(os.path.join(save_folder, f'{model_name}.sav'), 'wb'))
    return models


def load_discriminators(model_names, file_folder):
    models = dict()
    for model_name in model_names:
        loaded_model = pickle.load(open(os.path.join(file_folder, f'{model_name}.sav'), 'rb'))
        models[model_name] = loaded_model
    return models


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_or_load", type=str, default='train')
    parser.add_argument("--dataset", help="dataset name", type=str, choices=["mnist", "cifar10", "gtsrb"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_folder", type=str, default='./result/saved_models')
    parser.add_argument("--plot_folder", type=str, default='./result/detection/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if not os.path.isdir('models'):
        os.makedirs('models')

    attack_class = [('DoS', 0.0)]
    # attack_class = [('Probe', 2.0)]
    # attack_class = [('R2L', 3.0)]
    # attack_class = [('U2R', 4.0)]

    data = NSL_KDD(attack_class)

    # load data
    train_dataset = NSL_data(data.train_data, data.train_labels)
    test_dataset = NSL_data(data.test_data, data.test_labels)
    valid_dataset = NSL_data(data.validation_data, data.validation_labels)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # train and test the self-built MLP model
    # mlp_model = train_ids(data.input_shape, trainloader, validloader, testloader, save_folder=args.model_folder)
    # y_pred = mlp_predict(mlp_model, testloader)
    # y_pred = np.argmax(y_pred, axis=1)
    # present_split_acc(y_pred, data.test_labels)
    # split_evaluate(data.test_labels, y_pred[:, 1], plot=True, filename=args.plot_folder)

    # train other models including SVM LG BNB DT
    model_names = ['LGR', 'KNN', 'BNB', 'SVM', 'DTC', 'MLP']
    if 'train' in args.train_or_load:
        models = train_discriminators(data.train_data, data.train_labels, save_folder=args.model_folder)
    else:
        models = load_discriminators(model_names, file_folder=args.model_folder)

    for model_name in model_names:
        print('Model tested: {}'.format(model_name))
        model = models[model_name]
        y_pred = model.predict(data.test_data)
        present_split_acc(y_pred, data.test_labels)


    # test models including SVM LG BNB DT
    # for model_name in model_names:
        # scores = get_score(model_name, models[model_name], data.test_data)
        # split_evaluate(data.test_labels, scores, plot=True, filename=args.plot_folder+model_name)

    # performance = present_result(data.test_data, data.test_labels, models)





