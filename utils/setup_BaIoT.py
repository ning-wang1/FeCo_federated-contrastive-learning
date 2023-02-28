from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import os
import numpy as np
import itertools
import sys
import copy
from matplotlib import pyplot as plt
import seaborn as sns
import textwrap

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

# from scipy.stats import spearmanr
# from scipy.cluster import hierarchy
import scipy.stats
import scipy.cluster

FILE_PATHS = {'Danmini_Doorbell': 'dataset/IoTbot/Danmini_Doorbell',
              'Ecobee_Thermostat': 'dataset/IoTbot/Ecobee_Thermostat',
              'Ennio_Doorbell': 'dataset/IoTbot/Ennio_Doorbell',
              'Philips_B120N10_Baby_Monitor': 'dataset/IoTbot/Philips_B120N10_Baby_Monitor',
              'Provision_PT_737E_Security_Camera': 'dataset/IoTbot/Provision_PT_737E_Security_Camera',
              'Provision_PT_838_Security_Camera': 'dataset/IoTbot/Provision_PT_838_Security_Camera',
              'Samsung_SNH_1011_N_Webcam': 'dataset/IoTbot/Samsung_SNH_1011_N_Webcam',
              'SimpleHome_XCS7_1002_WHT_Security_Camera': 'dataset/IoTbot/SimpleHome_XCS7_1002_WHT_Security_Camera',
              'SimpleHome_XCS7_1003_WHT_Security_Camera': 'dataset/IoTbot/SimpleHome_XCS7_1003_WHT_Security_Camera'
              }
# Dir = os.path.split(os.getcwd())[0]
Dir = os.getcwd()
MIRAI = ['ack', 'scan', 'syn', 'udp', 'udpplain']
GAFGYT = ['combo', 'junk', 'scan', 'tcp', 'udp']
attack_label_dic = {'ack': 1, 'scan_mirai': 2, 'syn': 3, 'udp_mirai': 4, 'udpplain': 5,
                    'combo': 6, 'junk': 7, 'scan_gafgyt': 8, 'tcp': 9, 'udp_gafgyt': 10}


DEVICE_NAMES = ['Danmini_Doorbell', 'Ecobee_Thermostat',
               'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
               'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera',
               'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera',
               'SimpleHome_XCS7_1003_WHT_Security_Camera']


class BaIoT_data(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y
        self.length = x.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        record = self.data[idx, :]
        label = self.labels[idx]
        return record, label


class BaIoT:
    def __init__(self, iot_device):
        # np.random.seed(1)
        self.filepath = FILE_PATHS[iot_device]
        x_train_a, x_train_n, x_val_a, x_val_n, x_test_a, x_test_n = self.split_train_test(train=3)
        self.x_train_attack = x_train_a
        self.x_train_normal = x_train_n
        self.x_val_attack = x_val_a
        self.x_val_normal = x_val_n
        self.x_test_attack = x_test_a
        self.x_test_normal = x_test_n
        x_train, y_train, x_val, y_val, x_test, y_test = \
            self.build_data(x_train_a, x_train_n, x_val_a, x_val_n, x_test_a, x_test_n)
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def combine_attacks(self, mirai_attacks=None, gafgyt_attacks=None):
        """ Loading data
        """
        print('loading data ...')
        #  Load BaIoT dataset
        file_path = self.filepath
        gafgyt_num = 0
        mirai_num = 0

        if gafgyt_attacks is not None:
            for i, attack in enumerate(gafgyt_attacks):
                path_gafgyt = os.path.join(Dir, file_path, 'gafgyt_attacks/{}.csv'.format(attack))
                df_gafgyt = pd.read_csv(path_gafgyt, sep=",")  # load data
                if i == 0:
                    data_attack = df_gafgyt.to_numpy()
                else:
                    data_attack = np.concatenate([data_attack, df_gafgyt.to_numpy()], axis=0)

            gafgyt_num = len(data_attack)
            # print(f'num of gafgyt instance is {gafgyt_num}')

        if mirai_attacks is not None:
            for i, attack in enumerate(mirai_attacks):
                path_mirai = os.path.join(Dir, file_path, 'mirai_attacks/{}.csv'.format(attack))
                df_mirai = pd.read_csv(path_mirai, sep=",")  # load data
                if gafgyt_attacks is None and i == 0:
                    data_attack = df_mirai.to_numpy()
                else:
                    data_attack = np.concatenate([data_attack, df_mirai.to_numpy()], axis=0)
            mirai_num = len(data_attack)-gafgyt_num
            # print(f'num of Mirai instance is {len(data_attack)-gafgyt_num}')

        return data_attack, gafgyt_num, mirai_num

    def get_minmax(self, data):

        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        min_val = min_val.reshape(1, -1)
        max_val = max_val.reshape(1, -1)

        return min_val, max_val

    def scale_data(self, data_list):

        for i, data in enumerate(data_list):
            if i == 0:
                min_val, max_val = self.get_minmax(data)
            else:
                min_curr, max_curr = self.get_minmax(data)
                min_val = np.min(np.concatenate((min_val, min_curr), axis=0), axis=0)
                max_val = np.max(np.concatenate((max_val, max_curr), axis=0), axis=0)
                min_val = min_val.reshape(1, -1)
                max_val = max_val.reshape(1, -1)

        fea_num = data.shape[1]
        for data in data_list:
            for i in range(fea_num):
                i_col = data[:, i]
                i_col = (i_col - min_val[0, i]) / (max_val[0, i] - min_val[0, i])
                data[:, i] = i_col
        return data_list

    def split_train_test(self, train=3, test=2):

        print('split the data into training set and testing set ...')
        # load normal data
        file_path = self.filepath
        path_norm = os.path.join(Dir, file_path, 'benign_traffic.csv')
        df_norm = pd.read_csv(path_norm, sep=",")  # load data
        data_normal = df_norm.to_numpy()

        # load attack data split them to training set and test set
        if os.path.exists(os.path.join(Dir, file_path, 'mirai_attacks/')):
            x_train_attack, g_num_train, m_num_train = self.combine_attacks(MIRAI[0:train])
            x_test_attack, g_num_test, m_num_test = self.combine_attacks(MIRAI[train:], GAFGYT)
        else:
            x_train_attack, g_num_train, m_num_train = self.combine_attacks(gafgyt_attacks=GAFGYT[0:train])
            x_test_attack, g_num_test, m_num_test = self.combine_attacks(gafgyt_attacks=GAFGYT[train:])
        print(f'num of gafgyt instance is {g_num_train + g_num_test}')
        print(f'num of mirai instance is {m_num_train + m_num_test}')
        data_scaled = self.scale_data([data_normal, x_train_attack, x_test_attack])
        data_normal = data_scaled[0]
        x_train_attack = data_scaled[1]
        x_test_attack = data_scaled[2]

        # # train test ratio
        train_attack_len = x_train_attack.shape[0]
        test_attack_len = x_test_attack.shape[0]
        train_test_ratio = train_attack_len / (train_attack_len + test_attack_len)
        print('attack data for train (MIRAI: {}), attack data for test (Mirai and GAFGYT: {})'.format(
            train_attack_len, test_attack_len))

        # split the normal data
        normal_len = data_normal.shape[0]
        index = np.arange(normal_len)
        # np.random.shuffle(index)
        train_test_ratio = 0.7
        x_train_normal = data_normal[index[0: int(normal_len * train_test_ratio)], :]
        x_test_normal = data_normal[index[int(normal_len * train_test_ratio)]:, :]

        train_normal_whole_len = x_train_normal.shape[0]
        train_len = int(train_normal_whole_len * 0.7)
        x_val_normal = x_train_normal[train_len:, :]
        x_train_normal = x_train_normal[0:train_len, :]

        train_attack_whole_len = x_train_attack.shape[0]
        train_len = int(train_attack_whole_len * 0.9)
        x_val_attack = x_train_attack[train_len:, :]

        print('the length of normal data in training set is: {}'.format(x_train_normal.shape))
        print('the length of attack data in training set is {}'.format(x_train_attack.shape))
        print('the length of normal data in validation set is: {}'.format(x_val_normal.shape))
        print('the length of attack data in validation set is {}'.format(x_val_attack.shape))
        print('the length of normal data in testing set is: {}'.format(x_test_normal.shape))
        print('the length of attack data in testing set is {}'.format(x_test_attack.shape))
        return x_train_attack, x_train_normal, x_val_attack, x_val_normal, x_test_attack, x_test_normal

    def build_data(self, x_train_a, x_train_n, x_val_a, x_val_n, x_test_a, x_test_n):
        x_train = np.concatenate([x_train_a, x_train_n], axis=0)
        y_train = np.concatenate([np.ones(x_train_a.shape[0]), np.zeros(x_train_n.shape[0])])
        y_train = y_train.astype(int)

        x_test = np.concatenate([x_test_a, x_test_n], axis=0)
        y_test = np.concatenate([np.ones(x_test_a.shape[0]), np.zeros(x_test_n.shape[0])])
        y_test = y_test.astype(int)

        x_val = np.concatenate((x_val_a, x_val_n), axis=0)
        y_val = np.concatenate([np.ones(x_val_a.shape[0]), np.zeros(x_val_n.shape[0])])
        y_val = y_val.astype(int)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def dataset(self):
        x_train, y_train, x_val, y_val, x_test, y_test = self.x_train, self.y_train, self.x_val,\
                                                         self.y_val, self.x_test, self.y_test
        train_dataset = BaIoT_data(x_train, y_train)
        test_dataset = BaIoT_data(x_test, y_test)
        val_dataset = BaIoT_data(x_val, y_val)
        train_normal = self.x_train_normal
        train_dataset_normal = BaIoT_data(train_normal, np.zeros(train_normal.shape[0]))

        return train_dataset, train_dataset_normal, val_dataset, test_dataset

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader, DataLoader, DataLoader):
        train_set, train_set_normal, val_set, test_set = self.dataset()
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=0)

        train_loader_normal = DataLoader(dataset=train_set_normal, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=0)

        test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=0)

        valid_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=0)
        return train_loader, train_loader_normal, test_loader, valid_loader


if __name__ == '__main__':
    iot_device = 'Danmini_Doorbell'
    data = BaIoT(iot_device)
