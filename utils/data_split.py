
import numpy as np
from utils.setup_NSL import NSL_KDD, NSL_data


def get_dataset(args, all_data, normal_data, anormal_data):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_anormal_data = NSL_data(anormal_data.train_data, anormal_data.train_labels)
    train_normal_data = NSL_data(normal_data.train_data, normal_data.train_labels)
    test_data = NSL_data(all_data.test_data, all_data.test_labels)
    valid_data = NSL_data(all_data.validation_data, all_data.validation_labels)

    # Sample user data (data distribution among users are 'iid', 'non-iid', or 'attack-split')
    user_groups_normal = nsl_iid(train_normal_data, args.num_users)
    if args.data_distribution is 'iid':
        user_groups_anormal = nsl_iid(train_anormal_data, args.num_users)
    elif args.data_distribution is 'non-iid':
        user_groups_anormal = nsl_noniid(train_anormal_data, anormal_data.train_labels, args.num_users)
    elif args.data_distribution is 'attack-split:':
        user_groups_anormal = nsl_attack_split(train_anormal_data, anormal_data.train_labels, args.num_users)

    return train_normal_data, train_anormal_data, valid_data, test_data, user_groups_normal, user_groups_anormal


def nsl_iid(dataset, num_users):
    """
    Sample I.I.D. client data from nsl-kdd dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def nsl_noniid(dataset, labels, num_users):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param labels: the true labels of the dataset
    :param num_users:
    :return:
    """
    num_items = int(len(dataset) / num_users)

    num_shards = 100
    num_data = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_data)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, int(100/num_users), replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_data:(rand+1)*num_data]), axis=0)
    return dict_users


def nsl_attack_split(dataset, labels, num_users=4):
    """
    Sample non-I.I.D client data from dataset
    :param dataset:
    :param labels: the true labels of the dataset
    :param num_users:
    :return:
    """
    assert num_users == 4, "The number of users should be 4 to split the 4 attacks!"
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    attack_labels = [0, 2, 3, 4]
    for i, attack_label in enumerate(attack_labels):
        idx = np.where(labels == attack_label)
        dict_users[i] = idx

    return dict_users

