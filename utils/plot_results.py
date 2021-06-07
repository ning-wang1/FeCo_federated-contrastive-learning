import pandas as pd
import os
from matplotlib import pyplot as plt
from pylab import *
import random
from matplotlib.ticker import FormatStrFormatter
# import seaborn as sns
import global_vars as gv
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.mixture import GaussianMixture
import copy
from sklearn.metrics import recall_score


def str_2_float(str_ls, split_sym=',', strip_sym = '[]'):
    str_ls = str_ls.strip(strip_sym).split(split_sym)
    float_ls = [float(i) for i in str_ls]
    return float_ls


def str_2_int(str_ls, split_sym=',', strip_sym = '[]'):
    str_ls = str_ls.strip(strip_sym).split(split_sym)
    int_ls = [int(i) for i in str_ls]
    return int_ls


def plot_mmd(file_path):

    file_name = os.path.split(file_path)[-1]
    file_name_wo_suffix = file_name.split('.')[0]
    pp = PdfPages('figures/box_{}.pdf'.format(file_name_wo_suffix))
    fig1, ax = plt.subplots(figsize=(3.8, 3.5))
    # ax = plt.gca()

    data = pd.read_csv(file_path)
    agents = data['agents']
    mmd_ls = data['mmd']
    mal_agents = data['malicious_agents']
    detected_mal_agents = data['detected_mal_agents']
    mmd_dic = {}
    y_mal_ls =[]
    for t in range(1,len(agents)):
        curr_agents = str_2_int(agents[t], split_sym=' ')
        curr_mal_agents = str_2_int(mal_agents[t], split_sym=' ')
        curr_ben_agents = [i for i in curr_agents if i not in curr_mal_agents]

        mal_pos = [curr_agents.index(agent) for agent in curr_mal_agents]
        ben_pos = list(set(list(range(len(curr_agents)))) - set(mal_pos))

        y_mal = np.array(str_2_float(mmd_ls[t]))[mal_pos]
        y_ben = np.array(str_2_float(mmd_ls[t]))[ben_pos]

        mmd_dic[t] = y_ben
        y_mal_ls.append(y_mal)

        # plt.scatter(t * np.ones(len(y_mal)), y_mal, 60, 'r', '.', 'filled')
        # plt.scatter(t, y_mal, 60, 'r', '.', 'filled')

    # plt.scatter(t, y_mal, 60, 'r', '.', 'filled', label='Malicious Agent')
    colour = 'b'
    df = pd.DataFrame(mmd_dic)
    df.plot.box(title="Box plot of MMDs of benign agents",
                ax=ax, label='Benign Agents', color={'whiskers': colour,
                                 'caps': colour,
                                 'medians': colour,
                                 'boxes': colour})
    plt.scatter(np.arange(1, 20), y_mal_ls, 60, 'r', '.', 'filled')
    plt.xlabel('Training Iteration')
    plt.ylabel('MMD')
    plt.xticks(np.arange(1, 21, 2), labels=np.arange(1, 21, 2))
    plt.yticks(np.arange(0, 0.41, 0.1))

    legend_elements = [Line2D([0], [0], marker='o', color='w', markeredgecolor='k', label='Outlier of Benign Agents', markersize=8),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='r', label='Malicious Agent', markersize=8)]

    plt.legend(handles=legend_elements, loc='best')
    plt.tight_layout()
    pp.savefig()
    plt.show()
    plt.close()
    pp.close()


def average_recall_precision(file_path):
    data = pd.read_csv(file_path)
    recall = data['recall']
    precision = data['precision']
    fpr = data['fpr']

    row_num = len(recall)

    if 'distributed' in file_path:
        print(f'the average recall is {np.sum(recall)/row_num}')
        print(f'the average precision is {np.sum(precision) / row_num}')
        print(f'the average fpr is {np.sum(fpr) / row_num}\n')
    else:
        print(f'the final recall is {recall[row_num-1]}')
        print(f'the final precision is {precision[row_num-1]}')
        print(f'the fibal fpr is {fpr[row_num-1]} \n')


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
        tp += num_per_class[i] * r / 100
    fp = neg * fpr /100
    tn = neg - fp
    fn = pos - tp
    recall = tp/pos
    acc = (tp + tn)/(neg + pos)
    precision = tp / (tp + fp)
    f1 = recall * precision * 2 / (recall + precision)
    print(f'acc: {acc}, recall: {recall}, precision: {precision}, f1: {f1}')


def recall1_to_recall2(num_per_class, recall_per_class):
    total_num = np.sum(num_per_class)
    pass



if __name__ == "__main__":
    # file_dir = '/home/ning/extens/federated_contrastive/result/score/fl/'
    # filenames = ['metrics_non-iid_lr0.0001_clients50_seed1.csv',
    #              'metrics_non-iid_lr0.001_clients50_seed1.csv']
    # file_path = file_dir + filenames[1]
    # average_recall_precision(file_path)
    #
    # file_dir = '/home/ning/extens/federated_contrastive/result/score/distributed/'
    # file_path = file_dir + filenames[0]
    # average_recall_precision(file_path)

    # cave
    num_per_class = [7458, 2421, 2554, 400]
    recall_per_class = [87.7, 89.6, 44, 13.75]
    fpr = 8.18
    cal_metrics(num_per_class, recall_per_class, fpr)

    # icave
    num_per_class = [7458, 2421, 2754, 200]
    recall_per_class = [85.65, 74.97, 44.41, 11.00]
    fpr = 2.74
    cal_metrics(num_per_class, recall_per_class, fpr)

    # hfr
    num_per_class = [7458, 2421, 2754, 200]
    recall_per_class = [89.7, 80.2, 34.2, 29.5]
    fpr = 6.3
    cal_metrics(num_per_class, recall_per_class, fpr)

    # tdtc
    num_per_class = [7458, 2421, 2754, 200]
    recall_per_class = [88.2, 87.32, 42, 70.15]
    fpr = 5.57
    cal_metrics(num_per_class, recall_per_class, fpr)
    check_recall([94.43, 88.2, 87.32, 42, 70.15], 84.86)

    # two tier
    num_per_class = [7458, 2421, 2754, 200]
    recall_per_class = [85.29, 86.12, 36.06, 58.21]
    fpr = 5.44
    cal_metrics(num_per_class, recall_per_class, fpr)
    check_recall([94.56, 85.29, 86.12, 36.06, 58.21], 83.24)

    num_per_class = [7458, 2421, 2754, 200]
    recall_per_class = [83, 59, 10, 8]
    fpr = 3
    cal_metrics(num_per_class, recall_per_class, fpr)

    # define actual
    act_pos1 = [1 for _ in range(100)]
    act_pos2 = [2 for _ in range(100)]
    act_neg = [0 for _ in range(10000)]
    y_true = act_pos1 + act_pos2 + act_neg
    # define predictions
    pred_pos1 = [0 for _ in range(23)] + [1 for _ in range(70)] + [2 for _ in range(7)]
    pred_pos2 = [0 for _ in range(5)] + [2 for _ in range(95)]
    pred_neg = [0 for _ in range(10000)]
    y_pred = pred_pos1 + pred_pos2 + pred_neg
    # calculate recall
    recall = recall_score(y_true, y_pred, labels=[1,2], average='weighted')
    print('Recall: %.3f' % recall)

    # define actual
    act_pos = [1 for _ in range(100)]
    act_neg = [0 for _ in range(10000)]
    y_true = act_pos + act_neg
    # define predictions
    pred_pos = [0 for _ in range(10)] + [1 for _ in range(90)]
    pred_neg = [0 for _ in range(10000)]
    y_pred = pred_pos + pred_neg
    # calculate recall
    recall = recall_score(y_true, y_pred)
    print('Recall: %.3f' % recall)