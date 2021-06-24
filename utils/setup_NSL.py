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

# from keras.models import Sequential
# from keras.layers import Dense
from utils.classifier import evaluate_sub


FEATURE_NUM = 35  # the number of selected features
VALIDATION_SIZE = 5000
NORMAL_TRAIN_NUM = 20000
# file paths of training and testing data
# train_file_path = 'NSL_KDD/KDDTrain+_20Percent.txt'
# test_file_path = 'NSL_KDD/KDDTest-21.txt'

binary_col = ["land", "is_host_login", "is_guest_login", "logged_in", "root_shell"]
categorical_col = ['protocol_type', 'service', 'flag']

sys.path.append('../')
train_file_path = 'dataset/NSL_KDD/KDDTrain+.txt'
test_file_path = 'dataset/NSL_KDD/KDDTest+.txt'
SCALER = 'std_scale'
# SCALER = 'minmax'

# attributes/features of the data
datacols = ["duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"]

cols_cor = ["dst_host_srv_serror_rate", "srv_serror_rate", "serror_rate",
            "dst_host_serror_rate", "dst_bytes", "same_srv_rate",
            "src_bytes", "rerror_rate", "srv_rerror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "dst_host_count",
            "dst_host_srv_count", "logged_in", "dst_host_diff_srv_rate",
            "diff_srv_rate", "dst_host_srv_diff_host_rate", "dst_host_same_src_port_rate",
            "dst_host_same_srv_rate", "duration", "num_compromised",
            "hot", "is_guest_login", "num_access_files",
            "num_root", "su_attempted", "root_shell",
            "num_shells", "num_file_creations", "num_failed_logins",
            "urgent", "wrong_fragment", "land",
            "is_host_login", "srv_count", "count", "srv_diff_host_rate",
            "protocol_type", "service", "flag", "num_outbound_cmds"]

datacols_no_outbound = ["duration", "protocol_type", "service", "flag", "src_bytes",
                        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                        "num_file_creations", "num_shells", "num_access_files",
                        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                        "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]


class MinMax:
    def __init__(self, max_v, min_v):
        self.min_v = min_v
        self.max_v = max_v

    def transform(self, x):
        y = (x - self.min_v) / (self.max_v - self.min_v)
        return y

    def inverse_transform(self, y):
        x = y * (self.max_v - self.min_v) + self.min_v
        return x


def preprocessing(scaler_name=SCALER):
    """ Loading data
    """
    #  Load NSL_KDD train dataset
    df_train = pd.read_csv(train_file_path, sep=",", names=datacols)  # load data
    df_train = df_train.iloc[:, :-1]  # removes an unwanted extra field

    # Load NSL_KDD test dataset
    df_test = pd.read_csv(test_file_path, sep=",", names=datacols)
    df_test = df_test.iloc[:, :-1]

    # train set dimension
    print('Train set dimension: {} rows, {} columns'.format(df_train.shape[0], df_train.shape[1]))
    # test set dimension
    print('Test set dimension: {} rows, {} columns'.format(df_test.shape[0], df_test.shape[1]))

    datacols_range_continous = {"duration": 58329.0, "src_bytes": 1379963888.0, "dst_bytes": 1309937401.0,
                                "wrong_fragment": 3.0, "urgent": 14.0, "hot": 101.0, "num_failed_logins": 5.0,
                                "num_compromised": 7479.0, "num_root": 7468.0, "num_file_creations": 100.0,
                                "num_shells": 5.0,
                                "num_access_files": 9.0, "num_outbound_cmds": 0.0, "count": 511.0, "srv_count": 511.0,
                                "serror_rate": 1.0, "srv_serror_rate": 1.0, "rerror_rate": 1.0, "srv_rerror_rate": 1.0,
                                "same_srv_rate": 1.0, "diff_srv_rate": 1.0, "srv_diff_host_rate": 1.0,
                                "dst_host_count": 255.0,
                                "dst_host_srv_count": 255.0, "dst_host_same_srv_rate": 1.0,
                                "dst_host_diff_srv_rate": 1.0,
                                "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 1.0,
                                "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
                                "dst_host_rerror_rate": 1.0,
                                "dst_host_srv_rerror_rate": 1.0, "land": 1.0, "logged_in": 1.0, "root_shell": 1.0,
                                "su_attempted": 1.0, "is_host_login": 1.0, "is_guest_login": 1.0}

    datacols_range_discrere = {"land": 1, "logged_in": 1, "root_shell": 1, "su_attempted": 1, "is_host_login": 1,
                               "is_guest_login": 1}
    #  data preprocessing
    mapping = {'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'saint': 'Probe',
               'mscan': 'Probe',
               'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS', 'neptune': 'DoS', 'smurf': 'DoS',
               'mailbomb': 'DoS',
               'udpstorm': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
               'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R', 'xterm': 'U2R',
               'ps': 'U2R',
               'sqlattack': 'U2R', 'httptunnel': 'U2R',
               'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L',
               'imap': 'R2L',
               'spy': 'R2L', 'multihop': 'R2L', 'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L',
               'snmpgetattack': 'R2L',
               'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
               'normal': 'Normal'
               }
    attack_train = df_train['attack'].tolist()
    attack_test = df_test['attack'].tolist()
    print('\nAttacks in the training set')
    for attack_name in mapping.keys():
        if attack_name in attack_train:
            print(attack_name)
    print('\nAttacks in the testing Set')
    for attack_name in mapping.keys():
        if attack_name in attack_test:
            print(attack_name)

    # Apply attack class mappings to the dataset
    df_train['attack_class'] = df_train['attack'].apply(lambda v: mapping[v])
    df_test['attack_class'] = df_test['attack'].apply(lambda v: mapping[v])

    # Drop attack field from both train and test data
    df_train.drop(['attack'], axis=1, inplace=True)
    df_test.drop(['attack'], axis=1, inplace=True)

    # Attack Class Distribution
    attack_class_freq_train = df_train[['attack_class']].apply(lambda x: x.value_counts())
    attack_class_freq_test = df_test[['attack_class']].apply(lambda x: x.value_counts())
    attack_class_freq_train['frequency_percent_train'] = round(
        (100 * attack_class_freq_train / attack_class_freq_train.sum()), 2)
    attack_class_freq_test['frequency_percent_test'] = round(
        (100 * attack_class_freq_test / attack_class_freq_test.sum()), 2)

    cols = df_train.select_dtypes(include=['float64', 'int64']).columns
    cols_minus_binary = list(set(cols) - (set(cols) & set(binary_col)))

    if scaler_name is 'std_scale':
        # Scaling Numerical Attributes
        scaler = StandardScaler()
        # extract numerical attributes and scale it to have zero mean and unit variance
        scaler.fit(df_train[cols_minus_binary])
        train_numerical = scaler.transform(df_train[cols_minus_binary])
        test_numerical = scaler.transform(df_test[cols_minus_binary])

    else:
        # extract numerical attributes and scale it to have zero mean and unit variance
        cols = df_train.select_dtypes(include=['float64', 'int64']).columns
        min_max_scaler = MinMaxScaler()
        train_numerical = min_max_scaler.fit_transform(df_train[cols].values)
        test_numerical = min_max_scaler.transform(df_test[cols].values)

    # turn the result back to a data frame
    train_numerical_df = pd.DataFrame(train_numerical, columns=cols_minus_binary)
    test_numerical_df = pd.DataFrame(test_numerical, columns=cols_minus_binary)

    # cols with binary values
    scaler2 = StandardScaler()
    scaler2.fit(df_train[binary_col])
    train_binary = scaler2.transform(df_train[binary_col])
    test_binary = scaler2.transform(df_test[binary_col])
    train_binary_df = pd.DataFrame(train_binary, columns=binary_col)
    test_binary_df = pd.DataFrame(test_binary, columns=binary_col)

    # Encoding of categorical Attributes
    encoder = LabelEncoder()
    # extract categorical attributes from both training and test sets
    cat_train = df_train.select_dtypes(include=['object']).copy()
    cat_test = df_test.select_dtypes(include=['object']).copy()
    # encode the categorical attributes
    train_categorical = cat_train.apply(encoder.fit_transform)
    test_categorical = cat_test.apply(encoder.fit_transform)

    # data sampling
    # define columns and extract encoded train set for sampling
    x_train_categorical = train_categorical.drop(['attack_class'], axis=1)
    train = pd.concat([train_numerical_df, train_binary_df, x_train_categorical], axis=1)
    class_col = train.columns

    x_train = np.concatenate((train_numerical, train_binary_df.values, x_train_categorical.values), axis=1)
    x = x_train
    y_train = train_categorical[['attack_class']].copy()
    c, r = y_train.values.shape
    y = y_train.values.reshape(c, )

    # create test data frame
    test = pd.concat([test_numerical_df, test_binary_df, test_categorical], axis=1)
    test['attack_class'] = test['attack_class'].astype(np.float64)
    test['protocol_type'] = test['protocol_type'].astype(np.float64)
    test['flag'] = test['flag'].astype(np.float64)
    test['service'] = test['service'].astype(np.float64)

    print('Original dataset shape {}'.format(Counter(df_train)))
    print('Resampled dataset shape {}'.format(x.shape))

    return x, y, class_col, test, scaler


def rank_feas_by_correlation(x, col_name):

    """
    compute the correlations of different features and remove redundancy (i.e., only keep
    one of all the highly correlated features)
    """
    col_name_original = col_name
    continuous_and_ordinal_feas = [fea for fea in col_name_original if fea not in categorical_col]
    continuous_and_ordinal_idxes = [col_name_original.index(fea) for fea in continuous_and_ordinal_feas]
    x_original = copy.deepcopy(x)

    x = copy.deepcopy(x[:, continuous_and_ordinal_idxes])
    col_name = continuous_and_ordinal_feas

    corr, _ = scipy.stats.spearmanr(x)
    corr_linkage = scipy.cluster.hierarchy.ward(corr)

    # get the ids of the clustered features
    cluster_ids = scipy.cluster.hierarchy.fcluster(corr_linkage, 0, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    selected_fea_idxes = []

    selected_feas = [col_name[idx] for idx in selected_fea_idxes]

    new_features = selected_feas + categorical_col
    new_features_idxes = [col_name_original.index(fea) for fea in new_features]
    x = x_original[:, new_features_idxes]

    return x, new_features


def oneHot(train_data, test_data, features, col_names=('protocol_type', 'service', 'flag')):
    # translate the features to one hot
    enc = OneHotEncoder()
    train_data = train_data[features]
    test_data = test_data[features]
    category_max = [3, 70, 11]
    x_train_1hot = []
    x_test_1hot = []
    cat_num_dict = {}
    for col in col_names:
        print(col)
        if col in train_data.columns:  # split the columns to 2 set: one for numerical, another is categorical
            train_data_num = train_data.drop([col], axis=1)
            train_data_cat = train_data[[col]].copy()
            test_data_num = test_data.drop([col], axis=1)
            test_data_cat = test_data[[col]].copy()

            # Fit train data
            enc.fit(train_data_cat.append(test_data_cat))
            x_train_1hot.append(enc.transform(train_data_cat).toarray())
            x_test_1hot.append(enc.transform(test_data_cat).toarray())

            train_data = train_data_num
            test_data = test_data_num
            cat_num_dict[col] = x_train_1hot[-1].shape[1]

    x_train = train_data_num.values
    x_test = test_data_num.values

    for train_1hot, test_1hot in zip(x_train_1hot, x_test_1hot):
        x_train = np.concatenate((x_train, train_1hot), axis=1)
        x_test = np.concatenate((x_test, test_1hot), axis=1)
    return x_train, x_test, cat_num_dict


def cat_to_num(train_data, test_data, features, col_names=('protocol_type', 'service', 'flag')):
    # translate the features to one hot
    enc = OneHotEncoder()
    train_data = train_data[features]
    test_data = test_data[features]
    category_max = [3, 70, 11]
    x_train_1hot = []
    x_test_1hot = []

    for col in col_names:
        print(col)
        if col in train_data.columns:  # split the columns to 2 set: one for numerical, another is categorical
            train_data_num = train_data.drop([col], axis=1)
            train_data_cat = train_data[[col]].copy()
            test_data_num = test_data.drop([col], axis=1)
            test_data_cat = test_data[[col]].copy()

            # Fit train data
            enc.fit(train_data_cat.append(test_data_cat))
            x_train_1hot.append(enc.transform(train_data_cat).toarray())
            x_test_1hot.append(enc.transform(test_data_cat).toarray())

            train_data = train_data_num
            test_data = test_data_num

    x_train = train_data_num.values
    x_test = test_data_num.values

    for train_1hot, test_1hot in zip(x_train_1hot, x_test_1hot):
        x_train = np.concatenate((x_train, train_1hot), axis=1)
        x_test = np.concatenate((x_test, test_1hot), axis=1)
    return x_train, x_test


def data_partition(x, y, class_col, test, features, attack_class):
    """
    data partition to
    """
    attack_name = attack_class[0][0]
    new_col = list(class_col)
    new_col.append('attack_class')

    # add a dimension to target
    new_y = y[:, np.newaxis]

    # create a data frame from sampled data
    data_arr = np.concatenate((x, new_y), axis=1)
    data_df = pd.DataFrame(data_arr, columns=new_col)

    # x_train, x_test = oneHot(data_df, test, features)

    # create two-target classes (normal class and an attack class)
    class_dict = defaultdict(list)
    normal_class = [('Normal', 1.0)]

    class_dict = create_class_dict(class_dict, data_df, test, normal_class, attack_class)
    train_data = class_dict['Normal_' + attack_name][0]
    test_data = class_dict['Normal_' + attack_name][1]
    grpclass = 'Normal_' + attack_name

    # transform the selected features to one-hot
    x_train, x_test, cat_num_dict = oneHot(train_data, test_data, features)

    y_train = train_data[['attack_class']].copy()
    c, r = y_train.values.shape
    y_train = y_train.values.reshape(c, )
    y_test = test_data[['attack_class']].copy()
    c, r = y_test.values.shape
    y_test = y_test.values.reshape(c, )
    # transform the labels to one-hot

    y_train = 1 == y_train[:, None].astype(np.float32)
    y_test = 1 == y_test[:, None].astype(np.float32)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    print('x_train', x_train[0], x_train[0].shape)

    return x_train, x_test, y_train, y_test, cat_num_dict


def create_class_dict(class_dict, data_df, test_df, normal_class, attack_class):
    """ This function subdivides train and test dataset into two-class attack labels (normal and selected attack)
    return the loc of target attack and normal samples
    """
    j = normal_class[0][0]  # name of normal class
    k = normal_class[0][1]  # numerical representer of normal class
    i = attack_class[0][0]  # name of abnormal class(DOS, Probe, R2L, U2R)
    v = attack_class[0][1]  # numerical represent of abnormal classes 
    # [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]
    train_set = data_df.loc[(data_df['attack_class'] == k) | (data_df['attack_class'] == v)]

    # augmentation
    # if v == 4 or v == 3:
    #     for iter in range(10):
    #         train_set_aug = data_df.loc[data_df['attack_class'] == v]
    #         train_set = pd.concat([train_set, train_set_aug])

    class_dict[j + '_' + i].append(train_set)
    # test labels
    test_set = test_df.loc[(test_df['attack_class'] == k) | (test_df['attack_class'] == v)]
    class_dict[j + '_' + i].append(test_set)

    return class_dict


def get_two_classes_data(x, y, class_col, test_df, features):
    """ This function subdivides train and test dataset into two-class attack labels (either normal or attacks)
    return the loc of target attack and normal samples
    """
    new_col = list(class_col)
    new_col.append('attack_class')

    # add a dimension to target
    new_y = y[:, np.newaxis]

    # create a data frame from sampled data
    data_arr = np.concatenate((x, new_y), axis=1)
    train_df = pd.DataFrame(data_arr, columns=new_col)

    # transform the selected features to one-hot
    x_train, x_test, cat_num_dict = oneHot(train_df, test_df, features)

    y_train = train_df[['attack_class']].copy()
    c, r = y_train.values.shape
    y_train = y_train.values.reshape(c, )
    y_test = test_df[['attack_class']].copy()
    c, r = y_test.values.shape
    y_test = y_test.values.reshape(c, )

    y_train_multi_class = copy.deepcopy(y_train)
    y_test_multi_class = copy.deepcopy(y_test)

    # get binary labels
    y_train = y_train[:, None].astype(np.float32) == 1
    y_test = y_test[:, None].astype(np.float32) == 1

    y_train_multi_class = y_train_multi_class.astype(int)
    y_test_multi_class = y_test_multi_class.astype(int)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    # transform the labels to one-hot
    print('x_train', x_train[0], x_train[0].shape)

    return x_train, x_test, y_train, y_test, y_train_multi_class, y_test_multi_class, cat_num_dict


def create_class_dict_balance(class_dict, data_df, test_df, normal_class, attack_class):
    """ This function subdivides train and test dataset into two-class attack labels
    return the loc of target attack and normal samples
    """
    j = normal_class[0][0]  # name of normal class
    k = normal_class[0][1]  # numerical representer of normal class
    i = attack_class[0][0]  # name of abnormal class(DOS, Probe, R2L, U2R)
    v = attack_class[0][1]  # numerical represent of abnormal classes
    # [('DoS', 0.0), ('Probe', 2.0), ('R2L', 3.0), ('U2R', 4.0)]
    train_set_normal = data_df.loc[(data_df['attack_class'] == k)]
    train_set_anormal = data_df.loc[(data_df['attack_class'] == v)]
    df1 = train_set_normal.sample(frac=0.5)
    train_set = pd.concat([train_set_anormal, df1])
    class_dict[j + '_' + i].append(train_set)
    # test labels
    test_set = test_df.loc[(test_df['attack_class'] == k) | (test_df['attack_class'] == v)]
    class_dict[j + '_' + i].append(test_set)

    return class_dict


def feature_selection(x, y, x_col_name, FEATURE_NUM):
    """ feature selections
    """
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FEATURE SELECTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(x.shape)
    print(y.shape)
    rfc = RandomForestClassifier()
    # fit random forest classifier on the training set
    y = y.reshape(-1, 1)  # reshape the labels
    rfc.fit(x, y[:, 1])
    # extract important features
    score = np.round(rfc.feature_importances_, 3)
    significance = pd.DataFrame({'feature': x_col_name, 'importance': score})
    significance = significance.sort_values('importance', ascending=False).set_index('feature')
    # plot significance
    plt.rcParams['figure.figsize'] = (11, 4)
    significance.plot.bar()
    plt.show()

    # create the RFE model and select 10 attributes
    rfe = RFE(rfc, FEATURE_NUM)
    rfe = rfe.fit(x, y)

    # summarize the selection of the attributes
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), x_col_name)]
    selected_features = [v for i, v in feature_map if i == True]

    return selected_features


def variance_check(x_train, col_names, plot):
    x_train = abs(x_train)
    print(x_train.shape)
    var_filter = VarianceThreshold(threshold=0)
    train = var_filter.fit_transform(x_train)
    print(f'the variance of the features are {var_filter.variances_}')
    print(f'the features names are {col_names}')

    features = ['srv_rerror_rate', 'rerror_rate', 'serror_rate', 'dst_host_serror_rate', 'srv_serror_rate', 'dst_host_srv_serror_rate']
    idx = [list(col_names).index(col) for col in features]
    print(f'variance of {features} are: {var_filter.variances_[idx]}')

    # to get the count of features that are not constant
    a = var_filter.get_support()
    print(f'removed features are {[col_names[i] for i in range(x_train.shape[1]) if not var_filter.get_support()[i]]}')

    col_names = col_names[var_filter.get_support()]
    return train, np.array(var_filter.variances_[a]), col_names


def correlation_check(x, col_name, th, variances, plot):
    """
    compute the correlations of different features and remove redundancy (i.e., only keep
    one of all the highly correlated features)
    """
    col_name_original = list(col_name)
    continuous_and_ordinal_feas = [fea for fea in col_name_original if fea not in categorical_col]
    continuous_and_ordinal_idxes = [col_name_original.index(fea) for fea in continuous_and_ordinal_feas]
    x_original = copy.deepcopy(x)

    x = copy.deepcopy(x[:, continuous_and_ordinal_idxes])
    col_name = continuous_and_ordinal_feas

    corr, _ = scipy.stats.spearmanr(x)
    corr_linkage = scipy.cluster.hierarchy.ward(corr)
    col_idx = np.arange(len(col_name))

    # plot the hierarchy cluster
    fig, ax1 = plt.subplots(1, figsize=(4.5, 5))
    dendro = scipy.cluster.hierarchy.dendrogram(corr_linkage, labels=col_idx, ax=ax1, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro['ivl']))

    for i in range(len(col_name)):
        fig.text(0.975, 1-0.024*(i+1), str(i) + ': ' + col_name[i], ha='left')

    fig.tight_layout()
    plt.show()
    fig.savefig('/home/ning/extens/federated_contrastive/result/data_analytics/dendro.pdf', bbox_inches='tight')

    # plot the color map of the correlations
    fig, ax1 = plt.subplots(1, figsize=(6, 5))
    c = ax1.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])

    ax1.set_xticks(dendro_idx)
    ax1.set_yticks(dendro_idx)
    ax1.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax1.set_yticklabels(dendro['ivl'])

    plt.colorbar(c)

    plt.show()
    fig.savefig('/home/ning/extens/federated_contrastive/result/data_analytics/colormap.pdf', bbox_inches='tight')

    # get the ids of the clustered features
    cluster_ids = scipy.cluster.hierarchy.fcluster(corr_linkage, th, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # show the distribution of features in the same cluster
    colors = [['red', 'green'], ['orange', 'blue'], ['m', 'c']]
    selected_fea_idxes = []
    labels_all = []

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(6, 3), sharey=True)
        idx_plot = 0
        for v in cluster_id_to_feature_ids.values():

            if len(v) > 1:
                labels = [textwrap.fill(col_name[v[0]], 35), textwrap.fill(col_name[v[1]], 35)]
                labels_all = labels_all + labels
                sns.kdeplot(x[:, v[0]], ax=axs[idx_plot], shade=True, color=colors[idx_plot][0], label=labels[0], alpha=.2)
                sns.kdeplot(x[:, v[1]], ax=axs[idx_plot], shade=True, color=colors[idx_plot][1], label=labels[1], alpha=.2)

                box = axs[idx_plot].get_position()
                axs[idx_plot].set_position([box.x0, box.y0, box.width, box.height * 0.8])
                axs[idx_plot].legend(loc='center left', bbox_to_anchor=(-0.2, 1.12), ncol=1,
                                     handletextpad=0.1, handlelength=0.3)

                if variances[v[0]] > variances[v[1]]:
                    selected_fea_idxes.append(v[0])
                else:
                    selected_fea_idxes.append(v[1])
                idx_plot += 1
            else:
                selected_fea_idxes.append(v[0])

        plt.subplots_adjust(wspace=0, hspace=0)
        fig.text(0.5, 0.01, 'normalized feature value', ha='center')

        fig.tight_layout()
        plt.show()
        fig.savefig('/home/ning/extens/federated_contrastive/result/data_analytics/correlated_fea.pdf',
                    bbox_inches='tight')

    selected_feas = [col_name[idx] for idx in selected_fea_idxes]
    removed_features = set(col_name) - set(selected_feas)
    print(f'removed features are {removed_features}')

    # x = x[:, selected_features]

    new_features = [fea for fea in col_name_original if fea in selected_feas or fea in categorical_col]
    new_features_idxes = [col_name_original.index(fea) for fea in new_features]
    x = x_original[:, new_features_idxes]

    return x, new_features


def anova(X_train, y_train, col_name, k):
    """
    calculate the anoval scores of numerical features,
    return: the anova score and selected features (features with lower scores are removed)
    """
    fvalue_selector = SelectKBest(f_regression, k=k)  # select features with 20 best ANOVA F-Values
    fvalue_selector.fit_transform(X_train, y_train)
    mask = fvalue_selector.get_support()
    new_features = [col_name[i] for i in range(mask.shape[0]) if mask[i]]

    y = np.array(fvalue_selector.scores_)

    print(f'ANOVA test for numerical features,\n removed features ---\n{list(set(col_name) - set(new_features))}')

    return new_features, y


def mutual_information(x_train, y_train, col_name, k):
    """
    calculate the mutual information scores of categorical features,
    return: the mutual information score and selected features (features with lower scores are removed)
    """
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(x_train, y_train)
    # to get names of the selected features
    mask = selector.get_support()  # Output   array([False, False,  True,  True,  True, False ....])
    new_features = [col_name[i] for i in range(mask.shape[0]) if mask[i]]

    y = np.array(selector.scores_)
    print(f'Mutual information test for categorical features,\n removed features ---\n{list(set(col_name) - set(new_features))}')
    return new_features, y


def plot_pdf(data, y, title_str):
    fig = plt.figure(figsize=(10, 4))
    mal_data = data[y == 0]
    ben_data = data[y == 1]
    data_range = np.max(data) - np.min(data)
    all_data =[mal_data, ben_data]
    labels = ['Intrusion Traffic', 'Normal Traffic']
    if 'byte' in title_str:
        plt.hist(all_data, bins=20, range=[0, 2e4], histtype='bar', label=labels, align='left')
    else:
        plt.hist(all_data, bins=list(range(3)), histtype='bar', label=labels, align='left')
        plt.xticks([0, 1, 2])

    plt.legend(loc='best')
    plt.title(title_str)
    fig.tight_layout()
    plt.show()


def plot_ben_mal(col_name, new_features, x, y):
    """
    plot the feature distribution of normal traffic and intrusion traffic
    """
    removed_features = set(list(col_name)) - set(new_features)
    removed_features = ['urgent', 'num_compromised', 'num_root', 'su_attempted',
                        'land', 'is_host_login', 'root_shell', 'dst_bytes']
    y = y.astype(np.float32) == 1

    fig, axs = plt.subplots(2, 4, figsize=(6, 4), sharey=True)

    labels = ['Intrusion Traffic', 'Normal Traffic']
    for i, fea in enumerate(removed_features):
        row = int(np.floor(i/4))
        col = i%4
        x_fea = x[fea].values
        mal_data = x_fea[y == 0]
        ben_data = x_fea[y == 1]
        all_data = [mal_data, ben_data]

        if 'byte' in fea:
            axs[row, col].hist(all_data, bins=4, range=[0, 2e4], histtype='bar', label=labels, align='left')
        else:
            axs[row, col].hist(all_data, bins=list(range(3)), histtype='bar', label=labels, align='left')
            axs[row, col].set_xticks([0, 1, 2])

        if i==0 or i==4:
            axs[row, col].set_ylabel('count')

        axs[row, col].yaxis.get_major_formatter().set_powerlimits((0, 1))
        axs[row, col].set_title(fea)

    fig.text(0.5, 0.01, 'normalized feature value', ha='center')

    fig.subplots_adjust(left=0, bottom=0.3, right=0.5, top=1)
    fig.legend(labels=labels,  # The labels for each line
               borderaxespad=0.5,  # Small spacing around legend box
               loc='upper right', bbox_to_anchor=(0.75, 0), ncol=2)

    plt.show()

    fig.savefig('/home/ning/extens/federated_contrastive/result/data_analytics/removed_fea.pdf',
                bbox_inches='tight')


def plot_scores(anova_score, mutual_score, numerical_col_name, cat_col_name, col):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4))

    idx = np.argsort(mutual_score)

    ax1.bar([cat_col_name[i] for i in idx], mutual_score[idx], color='royalblue')
    ax1.set_xticks(np.arange(len(cat_col_name)))
    ax1.set_xticklabels([col.index(cat_col_name[i]) for i in idx], rotation=0)
    ax1.set_xlabel('(Categorical) feature index')
    ax1.set_ylabel('mutual_info score')

    idx = np.argsort(anova_score)

    ax2.bar([numerical_col_name[i] for i in idx], anova_score[idx], color='royalblue')
    ax2.set(yscale="log")
    ax2.set_xticks(np.arange(len(numerical_col_name)))
    ax2.set_xticklabels([col.index(numerical_col_name[i]) for i in idx], rotation=90)
    ax2.set_xlabel('(Numerical) feature index')
    ax2.set_ylabel('Anova score')

    plt.show()

    fig.savefig('/home/ning/extens/federated_contrastive/result/data_analytics/anova_mutual.pdf',
                bbox_inches='tight')


class NSL_KDD:
    def __init__(self, rng, attack_class=None, data_type=None, fea_selection=True):

        x, y, x_col_name, test, scaler = preprocessing()
        df_train = pd.read_csv(train_file_path, sep=",", names=datacols)  # load data
        df_train_original = df_train.iloc[:, :-1]  # removes an unwanted extra field

        if fea_selection:
            if os.path.exists('./dataset/NSL_KDD/selected_features.npy'):
                selected_features = np.load('./dataset/NSL_KDD/selected_features.npy', allow_pickle=True)
            else:
                # remove features with zero variance and correlated features
                # selected_features = feature_selection(x, y, x_col_name, FEATURE_NUM)
                x1, variances, x_col_name1 = variance_check(x, x_col_name, True)
                x2, x_col_name2 = correlation_check(x1, x_col_name1, th=0.2, variances=variances, plot=True)

                cate_features = binary_col + categorical_col
                l2 = len(cate_features)
                numerical_features = [fea for fea in x_col_name2 if fea not in cate_features]
                l1 = len(numerical_features)

                # for categorical features and numerical feature
                selected_features_categorical, mutual_score = mutual_information(x2[:, l1:], y, cate_features,
                                                                                 k=l2 - 3)
                selected_features_numerical, anova_score = anova(x2[:, 0: l1], y, numerical_features,
                                                                 k=l1 - 3)

                plot_ben_mal(numerical_features, selected_features_numerical, df_train_original, y)
                plot_scores(anova_score, mutual_score, numerical_features, cate_features, list(x_col_name1))
                selected_features = selected_features_numerical + selected_features_categorical

                # features = ['num_outbound_cmds', 'rerror_rate', 'serror_rate', 'dst_host_srv_serror_rate',
                #             'land', 'root_shell', 'is_host_login', 'num_compromised', 'num_root', 'dst_bytes',
                #             'su_attempted', 'urgent']
                #
                # num = 8
                # removed_features = [features[i] for i in range(num)]
                # selected_features = list(set(x_col_name) - set(removed_features))

                # rank the features based on the correlation
                selected_features = [fea for fea in cols_cor if fea in selected_features]

                np.save('./dataset/NSL_KDD/selected_features.npy', selected_features)
        else:
            selected_features = datacols_no_outbound

        print('selected_features -----------\n.', selected_features)
        print('removed features -------------\n', list(set(x_col_name) - set(selected_features)))

        # split the data
        if attack_class is None:
            x_train, x_test, y_train, y_test, y_train_multi_class, y_test_multi_class, cat_num_dict = \
                get_two_classes_data(x, y, x_col_name, test, selected_features)

            self.y_test_multi_class = y_test_multi_class
            y_train_multi_class = y_train_multi_class
        else:
            x_train, x_test, y_train, y_test, cat_num_dict = \
                data_partition(x, y, x_col_name, test, selected_features, attack_class)

        # train data shuffling
        train_len = x_train.shape[0]
        idx = list(range(train_len))
        rng.shuffle(idx)

        x_train = copy.deepcopy(x_train[idx, :])
        y_train = copy.deepcopy(y_train[idx])
        y_train_multi_class = copy.deepcopy(y_train_multi_class[idx])

        # select a subset
        if data_type == 'normal':
            idx_train = np.where(y_train == 1)
            idx_test = np.where(y_test == 1)

        elif data_type == 'anomaly':
            idx_train = np.where(y_train == 0)
            idx_test = np.where(y_test == 0)
        else:
            idx_train = list(np.arange(len(y_train)))
            idx_test = list(np.arange(len(y_test)))

        y_train = np.copy(y_train[idx_train])
        y_test = np.copy(y_test[idx_test])
        x_train = np.copy(x_train[idx_train])
        x_test = np.copy(x_test[idx_test])

        self.test_data = x_test
        self.test_labels = y_test
        self.cat_num_dict = cat_num_dict
        self.validation_data = x_train[:VALIDATION_SIZE, ]
        self.validation_labels = y_train[:VALIDATION_SIZE]
        self.train_data = x_train[VALIDATION_SIZE:, ]
        self.train_labels = y_train[VALIDATION_SIZE:]
        self.y_train_multi_class = y_train_multi_class[VALIDATION_SIZE:]
        self.y_valid_multi_class = y_train_multi_class[:VALIDATION_SIZE]

        self.scaler = scaler
        self.FEATURE_NUM_FINAL = x_train.shape[1]

        self.input_shape = x_train.shape[1]

    def data_rerange(self, data):
        scaler = MinMaxScaler()
        # extract numerical attributes and scale it to have zero mean and unit variance
        data = scaler.fit_transform(data) - 0.5
        return data

    def get_feature_mean(self):
        y = np.argmax(self.train_labels, axis=1)
        pos = np.where(y == 0)[0]
        x = self.train_data[pos, :37]
        x_inverse = self.scaler.inverse_transform(x)
        feature_mean_val = np.mean(x_inverse, axis=0)
        print(feature_mean_val.shape)
        print(feature_mean_val)
        return feature_mean_val


class NSL_data(Dataset):
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


class NSL_Dataset():
    def __init__(self, rng, normal_class=1, data_partition_type='normalOverAll'):
        super().__init__()

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [0]

        attack_type = {'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}
        if data_partition_type is "normalOverAll":
            data = NSL_KDD(rng)
            normal_data = NSL_KDD(rng, data_type='normal')
        else:
            attack = [(data_partition_type, attack_type[data_partition_type])]
            data = NSL_KDD(rng, attack)
            normal_data = NSL_KDD(rng, attack, data_type='normal')

        self.train_set = NSL_data(normal_data.train_data, normal_data.train_labels)
        self.valid_set = NSL_data(normal_data.validation_data, normal_data.validation_labels)
        self.test_set = NSL_data(data.test_data, data.test_labels)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=0)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=0)
        valid_loader = DataLoader(dataset=self.valid_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=0)
        return train_loader, test_loader, valid_loader

# class NSLModel:
#     def __init__(self, restore, feature_num, session=None):
#         self.num_features = feature_num
#         self.num_labels = 2
#
#         model = Sequential()
#         model.add(Dense(50, input_dim=feature_num, activation='relu'))
#         model.add(Dense(2))
#         model.load_weights(restore)
#         self.model = model
#
#     def predict(self, data):
#         return self.model(data)
#
#     def evaluate_only(self, x, y):
#         outputs = self.model(x).eval()
#         y_pred = np.argmax(outputs, axis=1)
#         evaluate_sub('nn model', y, y_pred)
#
#     def evaluate(self, x, y):
#         predicted = self.model(x).eval()
#         acc = np.count_nonzero(predicted.argmax(1) == y.argmax(1)) / y.shape[0]
#         return acc
