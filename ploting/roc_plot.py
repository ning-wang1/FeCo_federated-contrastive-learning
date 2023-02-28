import pandas as pd
import os
from matplotlib import pyplot as plt
from pylab import *
import random
from matplotlib.ticker import FormatStrFormatter
# import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import metrics
import pandas as pd

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.mixture import GaussianMixture
import copy
from sklearn.metrics import recall_score


def roc_plot(filenames):
    fig = plt.figure(figsize=(3.6, 3.3))
    colors = ['k', 'r', 'b']

    for i, filename in enumerate(filenames):
        label_score = np.load(filename)
        label = label_score[:, 0]
        score = label_score[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        name = os.path.split(filename)[-1]
        model_name = name.split('_')[0]
        if 'ae' in model_name:
            line_label = 'VAE'
        elif 'contrastive' in model_name:
            line_label = 'ConIDS'
        else:
            line_label = model_name

        line_label = line_label + ' (AUC={0:5.2f})'.format(auc)

        plt.plot(fpr, tpr, color=colors[i], label=line_label)

        # plt.fill_between(fpr, tpr, color='r', y2=0, alpha=0.3)
        # plt.tick_params(labelsize=23)
        # plt.text(0.9, 0.1, f'AUC: {round(AUC, 4)}', fontsize=25)
    plt.xticks(np.arange(0, 1.01, step=0.2))
    plt.xlabel('FPR')
    plt.ylabel('Recall')

    plt.xticks(np.arange(0,1.01,0.2))

    plt.legend()
    plt.show()
    fig.savefig('roc.pdf')


def fea_num():
    fea_num = [0, 1, 2, 3, 4, 5]
    acc = [0.8622, 0.8633, 0.8752, 0.8955, 0.8496, 0.8551]
    f1 = [0.8703, 0.873, 0.8859, 0.9044, 0.8620, 0.8683]

    fig = plt.figure(figsize=(3.6, 3.5))

    n = 5

    plt.plot(fea_num[:n], acc[:n], color='r', label='Accuracy', marker='s')
    plt.plot(fea_num[:n], f1[:n], color='b', label='F1 score', marker='o')
    plt.xticks(np.arange(0, 4.01, step=1))
    plt.xlabel('$K$')

    # plt.xlim([-0.01, 0.5])
    # plt.xticks(np.arange(0, 0.51, 0.1))
    plt.yticks(np.arange(0.75, 1.01, step=0.05))

    plt.legend()
    plt.show()
    fig.savefig('tune_fea.pdf')


def threshold_value():
    th = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    acc = [0.6678, 0.7552, 0.8100, 0.8643, 0.8824, 0.8955, 0.8927, 0.8836, 0.8834, 0.8830, 0.8822]
    recall = [0.4248, 0.5868, 0.6923, 0.7978, 0.8375, 0.8680, 0.8822, 0.8897, 0.8984, 0.9019, 0.9061]
    f1 = [0.5928, 0.7319, 0.8057, 0.8700, 0.8902, 0.9044, 0.9034, 0.8969, 0.8977, 0.8977, 0.8975]
    fpr = [0.0111, 0.0222, 0.0345, 0.0478, 0.0584, 0.0682, 0.0935, 0.1244, 0.1363, 0.1420, 0.1493]

    fig = plt.figure(figsize=(3.6, 3.5))

    n = 11

    plt.plot(th[:n], acc[:n], color='r', label='Accuracy', marker='s')
    plt.plot(th[:n], f1[:n], color='b', label='F1 score', marker='o')
    plt.plot(th[:n], recall[:n], color='g', label='Recall', marker='^')
    plt.plot(th[:n], fpr[:n], color='k', label='FPR', marker='+')

    plt.xticks(np.arange(0, 10.01, step=1))
    plt.yticks(np.arange(0, 1.01, step=0.2))
    plt.xlabel(r'$p$')
    plt.ylabel(' ')

    # plt.xlim([-0.01, 0.5])
    # plt.xticks(np.arange(0, 0.51, 0.1))

    plt.legend()
    plt.show()
    fig.savefig('tune_threshold.pdf')


def average_acc(filename):

    data = pd.read_csv(filename)

    col_names = ['acc', 'recall', 'precision', 'fpr', 'DoS', 'Probe', 'U2R', 'R2L']
    col_names = ['acc']
    for col in col_names:
        data_col = data[col]
        avg = np.mean(data_col)

        print(f'the average {col} is {avg}')


def f1(recall, precision):
    f1=recall*precision*2/(recall+precision)
    return f1


def converge(filenames):

    for filename in filenames:
        if 'non-iid' in filename:
            noniid1 = pd.read_csv(filename)
        elif 'attack-split' in filename:
            noniid2 = pd.read_csv(filename)
        else:
            iid = pd.read_csv(filename)
    n = 13
    col = 'recall'

    x = np.arange(n)
    plt.plot(x, iid[col], label='IID')
    plt.plot(x, noniid1[col], label='non-iid-1')
    plt.plot(x, noniid2[col], label='non-iid-2')

    # plt.plot(x, f1(iid['recall'], iid['precision']), label='IID')
    # plt.plot(x, f1(noniid1['recall'], noniid1['precision']), label='non-iid-1')
    # plt.plot(x, f1(noniid2['recall'], noniid2['precision']), label='non-iid-2')

    plt.ylim([0.6, 0.9])
    plt.legend()
    plt.show()


def convergence2(filenames):
    n=10
    fig = plt.figure(figsize=(3.6, 3.3))
    linestyles = ['-', '-.', '--']
    colors = ['k', 'b', 'r']
    for i, filename in enumerate(filenames):
        file = open(filename, 'r')
        if 'non-iid' in filename:
            label = 'non-iid-1'
            y0 = 2.54
        elif 'attack-split' in filename:
            label = 'non-iid-2'
            y0 = 2.19
        else:
            label = 'iid'
            y0 = 2.25

        lines = file.readlines()[1:]
        file.close()
        data = np.zeros([n, 4])
        for line in lines:
            record = line.split('\t')
            round_n = int(record[0])
            epoch = int(record[1])
            if round_n < n:
                if data[round_n, epoch] == 0:
                    data[round_n, epoch] = float(record[2])
                else:
                    data[round_n, epoch] += float(record[2])
                    # data[round_n, epoch] = float(record[2])
        data = data/5
        print(data)
        y = [y0] + list(data[:, -1])
        plt.plot(np.arange(len(y)), y, label=label, linestyle=linestyles[i], color=colors[i])

    plt.ylim([0.5, 2.5])
    plt.yticks(np.arange(0.5, 2.51, 0.5))
    plt.ylabel('Loss')
    plt.xlabel('Training Round')
    plt.legend()
    plt.show()
    fig.savefig('convergence.pdf')


def boxplot_acc(filenames):
    fig = plt.figure(figsize=(3.6, 3.3))
    for filename in filenames:
        if 'non-iid' in filename:
            noniid1 = pd.read_csv(filename)
        elif 'attack-split' in filename:
            noniid2 = pd.read_csv(filename)
        else:
            iid = pd.read_csv(filename)
    n = 13
    dic = dict()
    dic['iid'] = iid['acc']
    dic['non-iid-1'] = noniid1['acc']
    dic['non-iid-2'] = noniid2['acc']
    df = pd.DataFrame(dic)

    boxprops = dict(linestyle='-', linewidth=1, color='b')
    medianprops = dict(linestyle='-', linewidth=1, color='r')
    bp = df.boxplot(grid=False, rot=0, showfliers=False,
                    boxprops=boxprops, showmeans=True,
                    medianprops=medianprops,
                    return_type='dict')

    # boxplot style adjustments
    [item.set_linewidth(2) for item in bp['boxes']]
    [item.set_linewidth(2) for item in bp['fliers']]
    [item.set_linewidth(2) for item in bp['medians']]
    [item.set_linewidth(1) for item in bp['means']]
    [item.set_linewidth(2) for item in bp['whiskers']]
    [item.set_linestyle('--') for item in bp['whiskers']]
    [item.set_linewidth(2) for item in bp['caps']]

    [item.set_color('b') for item in bp['boxes']]
    # seems to have no effect
    [item.set_color('b') for item in bp['fliers']]
    [item.set_color('r') for item in bp['medians']]
    [item.set_markerfacecolor('r') for item in bp['means']]
    [item.set_marker('s') for item in bp['means']]
    [item.set_markeredgecolor('k') for item in bp['means']]
    [item.set_color('b') for item in bp['whiskers']]
    [item.set_color('k') for item in bp['caps']]

    patches = [mpatches.Patch(color='r', label="{:s}".format('mean'))]
    lengend_component = [Line2D([0], [0], marker='s', color='w', label='Mean',
                                markerfacecolor='r', markeredgecolor='k',
                                markersize=7),
                         Line2D([0], [0], color='r', linewidth=2, label='Median'),
                         Line2D([0], [0], marker='s', color='w', label='25%-75%',
                                markerfacecolor='w', markeredgecolor='b',
                                markersize=15, markeredgewidth=2)
                         ]
    plt.legend(handles=lengend_component, bbox_to_anchor=(0.46, 0.32), ncol=1)

    plt.xlabel('Data Distribution')
    plt.ylabel('Accuracy')
    plt.yticks(np.arange(0.4, 0.91, 0.1))
    plt.show()
    fig.savefig('boxplot.pdf')


def bar_plot(filename):
    fig = plt.figure(figsize=(3.6, 3.3))
    data = pd.read_csv(filename)

    x = ['DoS', 'Probe', 'R2L', 'U2R', 'All Data']
    y = data['acc']
    y = list(y) + [0.8955]
    plt.bar(x, y, width=0.5,  color='royalblue')
    plt.ylabel('Accuracy')
    plt.xlabel('Train Data')
    # plt.xticks(rotation=45)
    plt.yticks(np.arange(0, 0.91, 0.1))
    plt.show()
    fig.savefig('split_acc.pdf')


def client_num():
    fig = plt.figure(figsize=(3.6, 3.3))
    client_num = [10, 20, 30, 40, 50, 60]

    acc = [85.30, 84.63, 85.02, 86.39, 85.65, 84.29]
    acc = np.array(acc)/100
    plt.plot(client_num, acc, marker='o', label='FL', color='r')
    acc = [83.50, 83.18, 83.19, 82.07, 81.91, 81.62]
    acc = np.array(acc)/100
    plt.plot(client_num, acc, marker='s', label='Self-Learning', color='b')

    # plt.yticks([0, 20, 40, 60, 80, 100])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of Clients')
    plt.legend()
    plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    plt.show()
    fig.savefig('client_num.pdf')


def running_time():
    fig = plt.figure(figsize=(3.6, 3.3))
    seconds = [11.169, 5.557, 3.745, 2.819, 2.265, 1.878]
    total_n = 125973
    num_users = [10, 20, 30, 40, 50, 60]

    num_data = [total_n / num for num in num_users]
    plt.plot(num_users, seconds, marker='o', label='Per-client', color='b')
    plt.plot(num_users, [seconds[i]*num_users[i] for i in range(len(seconds))], marker='s', label='Total', color='r')
    plt.legend()
    plt.xlabel('Number of Clients')
    plt.ylabel('Training time (seconds)')
    plt.yticks(np.arange(0,121, 30))
    plt.show()
    fig.savefig('runtime.pdf')

    # fig = plt.figure(figsize=(3.8, 2.8))
    #
    # ax1 = fig.add_subplot(111)
    # ax1.plot(num_users, seconds, label='Train', linewidth=1.2, color='b')
    # # plt.yticks(np.arange(0.6, 1.01, 0.1))
    # plt.ylabel('Per Client Training Time')
    # plt.xlabel('Number of Clients')
    # ax1.xaxis.get_major_formatter().set_powerlimits((0, 2.0))
    # # plt.grid()
    # plt.legend(loc='lower right')
    #
    # ax2 = ax1.twinx()
    # ax2.plot(num_users, num_data, linestyle='--', linewidth=1.2, color='r')
    # # plt.ylim([0, 2 * 1e-6])
    # # ax2.yaxis.get_major_formatter().set_powerlimits((0, 10))
    # plt.ylabel('Data Size')
    # plt.show()


if __name__ == "__main__":
    dir = os.path.abspath(os.path.dirname(os.getcwd()))
    read_path = dir + '/result/0_ploting/'

    filenames = ['contrastive_labels_scores.npy','ae_labels_scores.npy', 'isoForest_labels_scores.npy']

    # roc_plot(filenames=[read_path+f for f in filenames])
    # fea_num()
    # threshold_value()
    #
    # # average_acc(read_path + 'distributed/metrics_iid_lr0.001_clients50_seed1.csv')
    # # average_acc(read_path + 'distributed/metrics_non-iid_lr0.001_clients50_seed1.csv')
    # # average_acc(read_path + 'distributed/metrics_attack-split_lr0.001_clients4_seed1.csv')
    #
    # for i in [10, 20, 30, 40, 50, 60]:
    #     average_acc(read_path + f'distributed/metrics_iid_lr0.001_clients{i}_seed1.csv')
    #
    # # for convergence
    # filenames = ['fl/metrics_attack-split_lr0.001_clients4_seed1_epochs52_le4_frac1.csv',
    #              'fl/metrics_iid_lr0.001_clients50_seed1_epochs52_le4_frac0.1.csv',
    #              'fl/metrics_non-iid_lr0.001_clients50_seed1_epochs52_le4_frac0.1.csv']
    # converge([read_path + filename for filename in filenames])
    #
    # filenames = [dir + '/logs/centralized/epoch.log']
    #
    # filenames = [f'/fl/epoch_iid.log',
    #              f'/fl/epoch_non-iid.log',
    #              f'/fl/epoch_attack-split.log']
    # convergence2([read_path + filename for filename in filenames])
    #
    filenames= ['distributed/metrics_iid_lr0.001_clients50_seed1.csv',
                'distributed/metrics_non-iid_lr0.001_clients50_seed1.csv',
                'distributed/metrics_attack-split_lr0.001_clients4_seed1.csv']

    # boxplot_acc([read_path + filename for filename in filenames])

    filenames = ['fl/old/metrics_iid_lr0.001_clients50_seed1_epochs52_le4_frac0.1.csv',
                 'fl/old/metrics_non-iid_lr0.001_clients50_seed1_epochs52_le4_frac0.1.csv',
                 'fl/old/metrics_attack-split_lr0.001_clients4_seed1_epochs52_le4_frac1.csv']
    boxplot_acc([read_path + filename for filename in filenames])
    #
    # bar_plot(read_path + 'distributed/metrics_attack-split_lr0.001_clients4_seed1.csv')
    #
    # client_num()
    # running_time()

