
import numpy as np
from utils.setup_NSL import NSL_KDD, NSL_data


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'nsl':
        all_data = NSL_KDD([('DoS', 0.0)], data_type=None)
        anormal_data = NSL_KDD([('DoS', 0.0)], data_type='anomaly')
        normal_data = NSL_KDD([('DoS', 0.0)], data_type='normal')

        train_anormal_data = NSL_data(anormal_data.train_data, anormal_data.train_labels)
        train_normal_data = NSL_data(normal_data.train_data, normal_data.train_labels)
        test_data = NSL_data(all_data.test_data, all_data.test_labels)
        valid_data = NSL_data(all_data.validation_data, all_data.validation_labels)

        # sample training data amongst users

        # Sample IID user data from Mnist
        user_groups_normal = nsl_iid(train_normal_data, args.num_users)
        user_groups_anormal = nsl_iid(train_anormal_data, args.num_users)

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


