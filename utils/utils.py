import csv
import numpy as np
import random
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.semi_supervised import LabelSpreading


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x ** 2, dim=dim).unsqueeze(dim))


def adjust_learning_rate(optimizer, lr_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_rate


def set_random_seed(manual_seed, use_cuda=False):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    rng = np.random.default_rng(manual_seed)
    torch.manual_seed(manual_seed)
    if use_cuda:
        torch.cuda.manual_seed(manual_seed)
    return rng


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


def split_evaluate(y, scores, plot=False, filename=None, manual_th=None, perform_dict=None):
    # compute FPR TPR
    pos_idx = np.where(y==1)
    neg_idx = np.where(y==0)
    score_pos = np.sum(scores[pos_idx])/len(pos_idx[0])
    score_neg = np.sum(scores[neg_idx])/len(neg_idx[0])

    if score_pos < score_neg:
        fpr, tpr, thresholds = roc_curve(y, -scores)
        thresholds = -thresholds
    else:
        fpr, tpr, thresholds = roc_curve(y, scores)
        thresholds = thresholds

    auc_score = auc(fpr, tpr)
    if plot:
        plot_roc(fpr, tpr, auc_score, filename)
    pos0 = np.where(fpr <= 0.02)[0]
    pos1 = np.where(fpr <= 0.04)[0]
    pos2 = np.where(fpr <= 0.06)[0]
    print(f'AUC: {auc_score}')
    print(f'TPR(FPR=0.02): {tpr[pos0[-1]]:.4f}, threshold: {thresholds[pos0[-1]]}\n')
    print(f'TPR(FPR=0.04): {tpr[pos1[-1]]:.4f}, threshold: {thresholds[pos1[-1]]}\n')
    print(f'TPR(FPR=0.06): {tpr[pos2[-1]]:.4f}, threshold: {thresholds[pos2[-1]]}\n')
    pos1 = np.where(fpr <= 0.08)[0]
    pos2 = np.where(fpr <= 0.1)[0]
    pos3 = np.where(fpr <= 0.12)[0]
    print(f'TPR(FPR=0.08): {tpr[pos1[-1]]:.4f}, threshold: {thresholds[pos1[-1]]}\n')
    print(f'TPR(FPR=0.1): {tpr[pos2[-1]]:.4f}, threshold: {thresholds[pos2[-1]]}\n')
    print(f'TPR(FPR=0.12): {tpr[pos3[-1]]:.4f}, threshold: {thresholds[pos3[-1]]}\n')

    # save scores to file
    labels_scores = np.concatenate((y.reshape(-1, 1), scores.reshape(-1, 1)), axis=1)
    if filename is not None:
        np.save(file=filename+'_labels_scores.npy', arr=labels_scores)

    # get the accuracy
    total_a = np.sum(y)
    total_n = len(y) - total_a
    best_acc = 0

    # evaluate the accuracy of normal set and anormal set separately using various threshold
    # acc = 0
    # total_correct_a = np.zeros(len(thresholds))
    # total_correct_n = np.zeros(len(thresholds))
    # if n_th > 500:
    #     thresholds_new = [thresholds[i] for i in range(n_th) if tpr[i]<1]

    # for i, th in enumerate(thresholds):
    #     if i % 500 == 0:
    #         print('evaluating threshold {}/{}'.format(i, len(thresholds)))
    #     y_pred = scores <= th
    #     correct = y_pred == y
    #     total_correct_a[i] += np.sum(correct[np.where(y == 1)])
    #     total_correct_n[i] += np.sum(correct[np.where(y == 0)])
    #
    # acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    # acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    # acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(thresholds))]
    # best_acc = np.max(acc)
    # idx = np.argmax(acc)
    # best_threshold = thresholds[idx]
    #
    # print('Best ACC: {:.4f} | Threshold: {:.4f} | ACC_normal={:.4f} | ACC_anormal={:.4f}\n'.
    #       format(best_acc, best_threshold, acc_n[idx], acc_a[idx]))

    if manual_th is not None:
        print(f'Manually choose decision threshold: {manual_th}')
    else:
        idxs = np.where(tpr >= 0.9998)
        id = idxs[0][0]
        manual_th = thresholds[id]
        print(f'choose decision threshold: {manual_th}')
    if score_pos < score_neg:
        y_pred = scores <= manual_th
    else:
        y_pred = scores >= manual_th
    correct = y_pred == y
    correct_a = np.sum(correct[np.where(y == 1)])
    correct_n = np.sum(correct[np.where(y == 0)])

    acc_n = correct_n / total_n
    acc_a = correct_a / total_a
    acc = (correct_n + correct_a) / (total_n + total_a)
    print('ACC: {:.4f} | Threshold: {:.4f} | ACC_normal={:.4f} | ACC_anormal={:.4f}\n'.
          format(acc, manual_th, acc_n, acc_a))

    recall = correct_a/total_a
    precision = correct_a/(total_n-correct_n+correct_a)
    fpr = (total_n - correct_n)/total_n

    print('Recall: {:.4f} | Precision: {:.4f} | fpr={:.4f} | f1={:.4f} \n'.
          format(recall, precision, fpr, 2*recall*precision/(recall+precision)))
    if perform_dict is not None:
        perform_dict['threshold'] = manual_th
        perform_dict['auc'] = auc_score
        perform_dict['acc'] = acc
        perform_dict['recall'] = recall
        perform_dict['precision'] = precision
        perform_dict['fpr'] = fpr

    return best_acc, acc, auc_score


def split_evaluate_w_label(y, y_pred, perform_dict=None):
    # compute FPR TPR

    if -1 in y:
        y = ((y + 1)/2).astype(int)

    if -1 in y_pred:
        y_pred = ((-y_pred + 1)/2).astype(int)

    # get the accuracy

    total_a = np.sum(y)
    total_n = len(y) - total_a

    # evaluate the accuracy of normal set and anormal set separately
    correct = y_pred == y
    correct_a = np.sum(correct[np.where(y == 1)])
    correct_n = np.sum(correct[np.where(y == 0)])

    acc_n = correct_n / total_n
    acc_a = correct_a / total_a
    acc = (correct_n + correct_a) / (total_n + total_a)

    print('ACC: {:.4f} |  ACC_normal={:.4f} | ACC_anormal={:.4f}\n'.
          format(acc, acc_n, acc_a))

    recall = correct_a/total_a
    precision = correct_a/(total_n-correct_n+correct_a)
    fpr = (total_n - correct_n)/total_n

    print('Recall: {:.4f} | Precision: {:.4f} | fpr={:.4f} | f1={:.4f} \n'.
          format(recall, precision, fpr, 2*recall*precision/(recall+precision)))
    if perform_dict is not None:
        perform_dict['acc'] = acc
        perform_dict['recall'] = recall
        perform_dict['precision'] = precision
        perform_dict['fpr'] = fpr
    return acc


def split_evaluate_two_steps(consist_pred, y, scores, manual_th=None, perform_dict=None):
    # compute FPR TPR
    total_a = 12833
    total_n = 9711
    # if the labels y is with 1, -1, then transform them to 1, 0
    if -1 in y:
        y = ((y + 1)/2).astype(int)

    y_pred = np.zeros(len(consist_pred))
    if manual_th is not None:
        print(f'Manually choose decision threshold: {manual_th}')
        idx1 = np.where(scores > manual_th)
        y_pred[idx1] = np.ones(len(idx1))

        idx2 = np.where(consist_pred[:, 1] > 0.966)
        idx22 = np.where(scores <= manual_th)

        idx3 = list(set(list(idx2[0])) & set(list(idx22[0])))
        y_pred[idx3] = np.ones(len(idx3))

        correct = y_pred == y
        correct_a = np.sum(correct[np.where(y == 0)])
        correct_n = np.sum(correct[np.where(y == 1)])

        acc = (correct_n + correct_a) / (len(y))

        recall = correct_a/total_a
        precision = correct_a/(total_n-correct_n+correct_a)
        fpr = (total_n - correct_n)/total_n

        print('Recall: {:.4f} | Precision: {:.4f} | fpr={:.4f} | f1={:.4f} \n'.
              format(recall, precision, fpr, 2*recall*precision/(recall+precision)))
        if perform_dict is not None:
            perform_dict['threshold'] = manual_th
            perform_dict['acc'] = acc
            perform_dict['recall'] = recall
            perform_dict['precision'] = precision
            perform_dict['fpr'] = fpr
    return acc


def per_class_acc(y, scores, manual_th, perform_dict=None):
    """ evaluate the prediction by showing per class accuracy
     param: y is the true label taking (0, 1, 2, 3, 4) where 'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0
     param: scores is the prediction scores
     param: manual_th: the manually selected threshold for decision making """

    # if not os.path.exists('./result/detection'):
    #     os.mkdir('./result/detection')

    print('The class wise accuracy')

    y_pred = (scores > manual_th).astype(int)
    fn_all = 0
    tp_all = 0

    # get the scores of normal data
    idxes = np.where(y == 1)
    y_pred_normal = y_pred[idxes]
    tn_all = np.sum(y_pred_normal)
    fp_all = len(y_pred_normal) - tn_all

    attacks = {'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}
    for attack_type, attack_id in attacks.items():
        idxes = np.where(y == attack_id)  # the index for ground truth attack
        y_pred_attack = y_pred[idxes]   # the prediction on ground-truth-attack
        fn = np.sum(y_pred_attack)
        tp = len(y_pred_attack) - fn
        recall = tp / (tp + fn)
        fn_all += fn
        tp_all += tp

        print(f'{attack_type} Attack, Recall: {recall:.4f}, normal: {fn}, intrusion: {tp}, total test num: {len(y_pred_attack)}')
        if perform_dict is not None:
            perform_dict[attack_type] = recall

    print(f'Overall recall {tp_all/(tp_all + fn_all)}')
    print(f'Noraml, normal: {tn_all}, intrusion: {fp_all}, total test num: {len(y_pred_normal)}')
    print(f'Normal traffic data, Recall: {tn_all/len(y_pred_normal):.4f}, Precision {tn_all/(tn_all + fn_all)}')


def get_threshold(scores, percent):
    threshold =np.percentile(scores, percent)
    return threshold


def plot_roc(fpr, tpr, auc_score, filename):
    f = plt.figure(figsize=(5, 3.5))
    plt.plot(fpr, tpr, color='r')
    # plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
    # plt.plot(np.array([0., 1.]), np.array([0., 1.]), color='b', linestyle='dashed')
    # plt.tick_params(labelsize=23)
    # plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
    # plt.xticks(np.arange(0, 1.01, step=0.2))
    # plt.xlim([-0.001, 0.011])
    # plt.xticks(np.arange(0, 0.01, step=0.001))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if 'contrastive' in filename:
        line_label = 'contrastive'
    else:
        line_label = os.path.split(filename)[-1]
    plt.legend([line_label + ' (AUC)={0:5.2f}'.format(auc_score), 'None'])
    plt.show()
    f.savefig(filename + '.pdf')


def check_recall(recall_ls, recall_all):
    data_num = [9711, 7458, 2421, 2754, 200]
    recall_all_compute = 0
    for idx, num in enumerate(data_num):
        recall_all_compute += num * recall_ls[idx]
    recall_all_compute = recall_all_compute/sum(data_num)
    print(f'the computed overall recall is {recall_all_compute}, the reported on is {recall_all} ')


def cal_metrics(num_per_class, recall_per_class, fpr):
    tp = 0
    neg = 9711
    pos = 12833
    for i, r in enumerate(recall_per_class):
        tp += num_per_class[i] * r
    fp = neg * fpr
    tn = neg - fp
    fn = pos - tp
    recall = tp/pos
    acc = (tp + tn)/(neg + pos)
    precision = tp / (tp + fp)
    f1 = recall * precision * 2 / (recall + precision)
    print(f'acc: {acc}, recall: {recall}, precision: {precision}, f1: {f1}')



