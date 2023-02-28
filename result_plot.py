import torch
import torch.backends.cudnn as cudnn
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


DEVICE_NAMES = ['Danmini_Doorbell', 'Ecobee_Thermostat',
               'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
               'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
               'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
               'SimpleHome_XCS7_1003_WHT_Security_Camera']


def cal_recall_old(recall_all_ls, recall_novel_ls):
    for i in range(len(recall_all_ls)):
        novel = 4412
        old = 9083
        all = 13495
        pp_all = recall_all_ls[i] * all
        pp_novel = recall_novel_ls[i] * novel
        recall_old = (pp_all-pp_novel)/old
        print('the recall of known attacks', recall_old)


def from_recall_fpr_to_all_metrics(recall, fpr):
    pos = 12833
    neg = 9711
    fp = neg * fpr
    tp = pos * recall
    fn = pos - tp
    tn = neg - fp
    precision = tp / (tp + fp)
    acc = (tp + tn) / (pos + neg)
    f1 = 2 * precision * recall / (precision + recall)

    print(f'Acc: {acc}, recall: {recall}, precision: {precision}, f1: {f1}, fpr: {fpr}')


def plot_fpr():
    methods = ['Autoencoder', 'SVM', 'IsolationForest', 'FeCo']
    FPR = {methods[0]: [0.0203, 0.0165, 0.0135],
           methods[1]: [0.0033, 0.0033, 0.0033],
           methods[2]: [0.0529, 0.0422, 0.0874],
           methods[3]: [0.0018, 0.0015, 0.0033]}

    df = pd.DataFrame(FPR)
    fig = plt.figure(figsize=(3.6, 3.3))

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
    # plt.yticks(np.arange(0.4, 0.91, 0.1))
    plt.show()
    fig.savefig('boxplot.pdf')

    # bar plot
    fig = plt.figure(figsize=(4.5, 3))
    y = []
    for m in methods:
        y.append(sum(FPR[m])/len(FPR[m]))
    plt.bar(methods, y, width=0.5, color='royalblue')
    plt.ylabel('FPR (Recall=0.9998)')
    plt.xlabel('Methods')
    # plt.xticks(rotation=45)
    # plt.yticks(np.arange(0, 0.91, 0.1))
    plt.show()
    fig.savefig('split_acc.pdf')

    # double bar plot
    # Set position of bar on X axis
    fig = plt.figure(figsize=(4.5, 3.3))
    barWidth = 0.25
    br1 = np.arange(len(y))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, y, color='r', width=barWidth,
            edgecolor='grey', label='IT')
    plt.bar(br2, 0.9998, color='g', width=barWidth,
            edgecolor='grey', label='ECE')

    # Adding Xticks
    plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(y))],
               methods)
    plt.show()
    fig.savefig('recall_fpr.pdf')


def distribution_plot(data):
    scores = data[:, 0]
    labels = data[:, 1]
    loc = np.where(labels==1)
    scores_pos = scores[loc]

    loc = np.where(labels==0)
    scores_neg = scores[loc]

    fig = plt.figure(figsize=(4.2,3.1))
    plt.subplot(211)
    sns.distplot(scores_pos, hist=True, kde=False,
                color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 2})
    # plt.legend(['Density', 'Intrusion traffic'], loc=9, bbox_to_anchor=(0.45, 0.95))
    plt.legend(['Intrusion traffic'], loc=9, bbox_to_anchor=(0.45, 0.95))
    plt.ylabel('Frequency')
    # plt.xlim([-1.05,1.05])

    plt.subplot(212)
    sns.distplot(scores_neg, hist=True, kde=False, color='red',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 2})
    plt.xlabel('Cosine similarity with normal template')
    # plt.xlim([-1.05, 1.05])
    # plt.legend(['Density', 'Benign traffic'], loc=9, bbox_to_anchor=(0.45, 0.95))
    plt.legend(['Benign traffic'], loc=9, bbox_to_anchor=(0.45, 0.95))
    plt.ylabel('Frequency')
    plt.show()
    fig.savefig(os.path.join(dir, f'ploting/score_dist_{device_name}.pdf'), bbox_inches='tight', dpi=1200)


def bar_plot_fpr():
    methods = ['Autoencoder', 'SVM', 'IsolationForest', 'FeCo']
    fpr = {}
    names = ['Device 1', 'Device 2', 'Device 3','Device 4','Device 5','Device 6',
             'Device 7', 'Device 8','Device 9']
    fpr['Device 1'] = {methods[0]: [0.0126, 0.0190, 0.0207],
                       methods[1]: [0.0017, 0.0017, 0.0017],
                       methods[2]: [0.0102, 0.0035, 0.0132],
                       methods[3]: [0.0016, 0.0016, 0.0017]}

    fpr['Device 3'] = {methods[0]: [0.0203, 0.0165, 0.0135],
                       methods[1]: [0.0033, 0.0033, 0.0033],
                       methods[2]: [0.0529, 0.0422, 0.0874],
                       methods[3]: [0.0015, 0.0015, 0.0013]}

    fpr['Device 2'] = {methods[0]: [0.022, 0.019, 0.019],
                       methods[1]: [0.047, 0.047, 0.047],
                       methods[2]: [0.0313, 0.0304, 0.0861],
                       methods[3]: [0.0077, 0.0077, 0.0078]}

    fpr['Device 4'] = {methods[0]: [0.0369, 0.0359, 0.0315],
                       methods[1]: [0.0565, 0.0565, 0.0565],
                       methods[2]: [0.0768, 0.0576, 0.0908],
                       methods[3]: [0.0037, 0.0037, 0.0037]}

    fpr['Device 5'] = {methods[0]: [0.0135, 0.055, 0.0479],
                       methods[1]: [0.005, 0.005, 0.005],
                       methods[2]: [0.2482, 0.2532, 0.2532],
                       methods[3]: [0.005, 0.005, 0.005]}

    fpr['Device 6'] = {methods[0]: [0.0278, 0.0284, 0.0257],
                       methods[1]: [0.0547, 0.0547, 0.0547],
                       methods[2]: [0.3275, 0.3301, 0.3303],
                       methods[3]: [0.0034, 0.0034, 0.0034]}

    fpr['Device 7'] = {methods[0]: [0.018, 0.018, 0.025],
                       methods[1]: [0.0462, 0.0462, 0.0462],
                       methods[2]: [0.0939, 0.0997, 0.0957],
                       methods[3]: [0.0037, 0.0037, 0.0038]}

    fpr['Device 8'] = {methods[0]: [0.0119, 0.0121, 0.0119],
                       methods[1]: [0.0051, 0.0051, 0.0051],
                       methods[2]: [0.0270, 0.0276, 0.0275],
                       methods[3]: [0.0031, 0.0031, 0.0031]}

    fpr['Device 9'] = {methods[0]: [0.0282, 0.0290, 0.0292],
                       methods[1]: [0.001, 0.001, 0.001],
                       methods[2]: [0.0575, 0.0584, 0.0624],
                       methods[3]: [0.0007, 0.0007, 0.0008]}



    # Make the plot
    fig = plt.figure(figsize=(6, 4))
    barWidth = 0.2
    br1 = np.arange(9)
    br2 = [x + barWidth for x in br1]
    br3 = [x + 2*barWidth for x in br1]
    br4 = [x + 3*barWidth for x in br1]

    plt.bar(br1, [sum(fpr[d][methods[2]]) / 3 for d in names], color='r', width=barWidth,
            edgecolor='r', label=methods[2])
    plt.bar(br2, [sum(fpr[d][methods[1]]) / 3 for d in names], color='b', width=barWidth,
            edgecolor='b', label=methods[1])
    plt.bar(br3, [sum(fpr[d][methods[0]]) / 3 for d in names], color='g', width=barWidth,
            edgecolor='g', label=methods[0])
    plt.bar(br4, [sum(fpr[d][methods[3]]) / 3 for d in names], color='k', width=barWidth,
            edgecolor='k', label=methods[3])

    # Adding Xticks
    # plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('FPR')
    plt.xticks([r + 1.5*barWidth for r in range(9)],
               names, rotation = 35)
    plt.legend()
    plt.show()
    fig.savefig('recall_fpr.pdf')


def bar_plot_accuracy():
    methods = ['Original Data (FedAvg)', 'Original Data (Krum)', 'Original Data (Coomed)',
               'Original Data (Tmean)', 'Original Data (Bylyan)', 'Original Data (FLTrust)',
               'PLR (FLARE)']
    fpr = {}
    names = ['fMNIST', 'CIFAR-10', 'KATHER']
    fpr['fMNIST'] = {methods[0]: 8.3,
                       methods[1]: 98.3,
                       methods[2]: 45,
                       methods[3]: 43.3,
                       methods[4]: 71.6,
                       methods[5]: 40,
                       methods[6]: 0
                       }
    fpr['CIFAR-10'] = {methods[0]: 26.7,
                       methods[1]: 100,
                       methods[2]: 53.3,
                       methods[3]: 33.3,
                       methods[4]: 65.0,
                       methods[5]: 0,
                       methods[6]: 0
                       }

    fpr['KATHER'] = {methods[0]: 41.6,
                       methods[1]: 86.7,
                       methods[2]: 33.3,
                       methods[3]: 41.7,
                       methods[4]: 56.7,
                       methods[5]: 19.4,
                       methods[6]: 0

                       }


    # Make the plot
    fig = plt.figure(figsize=(6, 3))
    barWidth = 0.1
    br1 = np.arange(3)
    br2 = [x + barWidth for x in br1]
    br3 = [x + 2*barWidth for x in br1]
    br4 = [x + 3*barWidth for x in br1]
    br5 = [x + 4*barWidth for x in br1]
    br6 = [x + 5 * barWidth for x in br1]
    br7 = [x + 6 * barWidth for x in br1]

    plt.bar(br1, [fpr[d][methods[0]] for d in names], color='red', width=barWidth,
            edgecolor='r', label=methods[0])
    plt.bar(br2, [fpr[d][methods[1]] for d in names], color='blue', width=barWidth,
            edgecolor='b', label=methods[1])

    plt.bar(br3, [fpr[d][methods[6]] for d in names], color='k', width=barWidth,
            edgecolor='k', label=methods[6])
    plt.bar(br4, [fpr[d][methods[3]] for d in names], color='y', width=barWidth,
            edgecolor='y', label=methods[3])

    plt.bar(br5, [fpr[d][methods[4]] for d in names], color='c', width=barWidth,
            edgecolor='c', label=methods[4])
    plt.bar(br6, [fpr[d][methods[5]] for d in names], color='m', width=barWidth,
            edgecolor='m', label=methods[5])

    plt.bar(br7, [fpr[d][methods[2]] for d in names], color='g', width=barWidth,
            edgecolor='g', label=methods[2])
    # plt.plot(br7, [fpr[d][methods[6]] for d in names], color='g', label=methods[6])




    # Adding Xticks
    # plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('Attack Success Rate')
    plt.ylim([-1,110])
    plt.xticks([r + 1.5*barWidth for r in range(3)],
               names, rotation = 0)
    plt.legend()
    plt.show()
    fig.savefig('recall_fpr.pdf')


if __name__ == '__main__':
    dir = os.getcwd()
    # for i in range(9):
    #     device_name = DEVICE_NAMES[i]
    #     score_file= os.path.join(dir, f'result/score/centralized/score_label_{device_name}.npy')
    #     score_label = np.load(score_file)
    #     distribution_plot(score_label)
    # plot_fpr()
    # bar_plot_fpr()
    bar_plot_accuracy()