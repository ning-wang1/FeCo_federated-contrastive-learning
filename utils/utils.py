import csv
import numpy as np
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x ** 2, dim=dim).unsqueeze(dim))


def adjust_learning_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate


class Logger(object):
    """Logger object for training process, supporting resume training"""

    def __init__(self, path, header, resume=False):
        """
        :param path: logging file path
        :param header: a list of tags for values to track
        :param resume: a flag controling whether to create a new
        file or continue recording after the latest step
        """
        self.log_file = None
        self.resume = resume
        self.header = header
        if not self.resume:
            self.log_file = open(path, 'w')
            self.logger = csv.writer(self.log_file, delimiter='\t')
            self.logger.writerow(self.header)
        else:
            self.log_file = open(path, 'a+')
            self.log_file.seek(0, os.SEEK_SET)
            reader = csv.reader(self.log_file, delimiter='\t')
            self.header = next(reader)
            # move back to the end of file
            self.log_file.seek(0, os.SEEK_END)
            self.logger = csv.writer(self.log_file, delimiter='\t')

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for tag in self.header:
            assert tag in values, 'Please give the right value as defined'
            write_values.append(values[tag])
        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]
    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1 * motion_length,) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name
    # replace the first convlution layer
    setattr(container, layer_name, new_conv)
    return base_model


def get_fusion_label(csv_path):
    """
    Read the csv file and return labels
    :param csv_path: path of csv file
    :return: ground truth labels
    """
    gt = np.zeros(360000)
    base = -10000
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[-1] == '':
                continue
            if row[1] != '':
                base += 10000
            if row[4] == 'N':
                gt[base + int(row[2]):base + int(row[3]) + 1] = 1
            else:
                continue
    return gt


def evaluate(score, label, whether_plot, model_id):
    """
    Compute Accuracy as well as AUC by evaluating the scores
    :param score: scores of each frame in videos which are computed as the cosine similarity between encoded test vector and mean vector of normal driving
    :param label: ground truth
    :param whether_plot: whether plot the AUC curve
    :return: best accuracy, corresponding threshold, AUC
    """
    thresholds = np.arange(0., 1., 0.01)
    best_acc = 0.
    best_threshold = 0.
    for threshold in thresholds:
        prediction = score >= threshold
        correct = prediction == label

        acc = (np.sum(correct) / correct.shape[0] * 100)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)

    if whether_plot:
        f = plt.figure(figsize=(5, 3.5))
        plt.plot(fpr, tpr, color='r')
        # plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
        plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
        # plt.tick_params(labelsize=23)
        # plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
        plt.xticks(np.arange(0, 1.01, step=0.2))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        f.savefig("{}.pdf".format(model_id), bbox_inches='tight')
    return best_acc, best_threshold, AUC


def post_process(score, window_size=6):
    """
    post process the score
    :param score: scores of each frame in videos
    :param window_size: window size
    :param momentum: momentum factor
    :return: post processed score
    """
    processed_score = np.zeros(score.shape)
    for i in range(0, len(score)):
        processed_score[i] = np.mean(score[max(0, i - window_size + 1):i + 1])

    return processed_score


def get_score(score_folder):
    """
    !!!Be used only when scores exist!!!
    Get the corresponding scores according to requiements
    :param score_folder: the folder where the scores are saved
    :param mode: top_d | top_ir | front_d | front_ir | fusion_top | fusion_front | fusion_d | fusion_ir | fusion_all
    :return: the corresponding scores according to requirements
    """

    score = np.load(os.path.join(score_folder + '/score.npy'))

    return score


def split_evaluate(y, scores, plot, filename, manual_th=None):
    if not os.path.exists('./result/detection'):
        os.mkdir('./result/detection')
    # compute FPR TPR
    print("    detailed detection results ----------------->")
    fpr, tpr, thresholds = roc_curve(y, scores)
    # compute auc score
    auc_score = auc(fpr, tpr)
    if plot:
        plot_roc(fpr, tpr, auc_score, filename)

    print('    AUC test: {:.3f}'.format(auc_score))
    pos = np.where(fpr <= 0.05)[0]
    print('    Set FPR=0.05, TPR={:.3f}'.format(tpr[pos[-1]]))
    pos = np.where(fpr <= 0.1)[0]
    print('    Set FPR=0.1, TPR={:.3f}'.format(tpr[pos[-1]]))

    # if the labels y is with 1, -1, then transform them to 1, 0
    if -1 in y:
        y = ((y + 1)/2).astype(int)

    # save scores to file
    labels_scores = np.concatenate((y.reshape(-1, 1), scores.reshape(-1, 1)), axis=1)
    np.save(file=filename+'_labels_scores.npy', arr=labels_scores)

    # get the accuracy
    total_correct_a = np.zeros(len(thresholds))
    total_correct_n = np.zeros(len(thresholds))
    total_n = np.sum(y)
    total_a = len(y) - total_n

    # evaluate the accuracy of normal set and anormal set separately using different threshold
    for i, th in enumerate(thresholds):
        # if i % 500 == 0:
        #     print('evaluating threshold {}/{}'.format(i, len(thresholds)))
        y_pred = scores > th
        correct = y_pred == y
        total_correct_a[i] += np.sum(correct[np.where(y == 0)])
        total_correct_n[i] += np.sum(correct[np.where(y == 1)])

    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(thresholds))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)
    best_threshold = thresholds[idx]

    print('    Best Accuracy: {:.3f}, Corresponding Threshold: {:.3f}'.format(best_acc, best_threshold))
    print('    Split Acc under the Best Acc: normal_acc={:.3f}, anormal_acc={:.3f}'.format(acc_n[idx], acc_a[idx]))

    if manual_th is not None:
        print(f'manually set threshold is: {manual_th}')
        y_pred = scores > manual_th
        correct = y_pred == y
        correct_a = np.sum(correct[np.where(y == 0)])
        correct_n = np.sum(correct[np.where(y == 1)])

        acc_n = correct_n / total_n
        acc_a = correct_a / total_a
        acc = (correct_n + correct_a) / (total_n + total_a)
        print('    Accuracy: {:.3f}, Corresponding Threshold: {:.3f}'.format(acc, manual_th))
        print('    Split Acc under the Best Acc: normal_acc={:.3f}, anormal_acc={:.3f}'.format(acc_n, acc_a))

    return best_acc, auc_score


def get_threshold(scores, percent):
    threshold =np.percentile(scores, percent)
    return threshold


def plot_roc(fpr, tpr, auc_score, filename):
    f = plt.figure(figsize=(5, 3.5))
    plt.plot(fpr, tpr, color='r')
    # plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
    plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
    # plt.tick_params(labelsize=23)
    # plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
    plt.xticks(np.arange(0, 1.01, step=0.2))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend([os.path.split(filename)[-1] + ' (AUC)={0:5.2f}'.format(auc_score), 'None'])
    plt.show()
    f.savefig(filename + '.pdf')

