import torch
from utils.utils import l2_normalize
import numpy as np
import os
from models.iso_forest import IsoForest


def get_normal_vector(model, train_normal_loader_for_test, cal_vec_batch_size, latent_dim, use_cuda):
    total_batch = int(len(train_normal_loader_for_test))
    print("------------------- Calculating Average Normal Vector --------------------")
    if use_cuda:
        normal_vec = torch.zeros((1, latent_dim)).cuda()
    else:
        normal_vec = torch.zeros((1, latent_dim))
    for batch, (normal_data, idx) in enumerate(train_normal_loader_for_test):
        if use_cuda:
            normal_data = normal_data.cuda()
        _, outputs = model(normal_data.float())
        outputs = outputs.detach()
        normal_vec = (torch.sum(outputs, dim=0) + normal_vec * batch * cal_vec_batch_size) / (
                (batch + 1) * cal_vec_batch_size)
        if (batch + 1) % 1000 == 0:
            print(f'Calculating Average Normal Vector: Batch {batch + 1} / {total_batch}')
    normal_vec = l2_normalize(normal_vec)
    return normal_vec


def split_acc_diff_threshold(model, normal_vec, test_loader, use_cuda):
    """
    Search the threshold that split the scores the best and calculate the corresponding accuracy
    """
    total_batch = int(len(test_loader))
    total_n = 0
    total_a = 0
    threshold = np.arange(0., 1., 0.01)
    total_correct_a = np.zeros(threshold.shape[0])
    total_correct_n = np.zeros(threshold.shape[0])
    for batch, batch_data in enumerate(test_loader):
        if use_cuda:
            batch_data[0] = batch_data[0].cuda()
            batch_data[1] = batch_data[1].cuda()
        n_num = torch.sum(batch_data[1]).cpu().detach().numpy()
        total_n += n_num
        total_a += (batch_data[0].size(0) - n_num)
        _, outputs = model(batch_data[0].float())
        outputs = outputs.detach()
        similarity = torch.mm(outputs, normal_vec.t())
        for i in range(len(threshold)):
            # If similarity between sample and average normal vector is smaller than threshold,
            # then this sample is predicted as anormal driving which is set to 0
            prediction = similarity >= threshold[i]
            correct = prediction.squeeze() == batch_data[1]
            total_correct_a[i] += torch.sum(correct[~batch_data[1].bool()])
            total_correct_n[i] += torch.sum(correct[batch_data[1].bool()])
        if (batch+1) % 100 == 0:
            print(f'Evaluating: Batch {batch + 1} / {total_batch} \n')

    acc_n = [(correct_n / total_n) for correct_n in total_correct_n]
    acc_a = [(correct_a / total_a) for correct_a in total_correct_a]
    acc = [((total_correct_n[i] + total_correct_a[i]) / (total_n + total_a)) for i in range(len(threshold))]
    best_acc = np.max(acc)
    idx = np.argmax(acc)
    best_threshold = idx * 0.01
    return best_acc, best_threshold, acc_n[idx], acc_a[idx], acc, acc_n, acc_a


def cal_score(model, normal_vec, test_loader, score_folder=None, use_cuda=False):
    """
    Generate and save scores
    """
    total_batch = int(len(test_loader))
    sim_1_list = torch.zeros(0)
    label_list = torch.zeros(0).type(torch.LongTensor)
    for batch, data1 in enumerate(test_loader):
        if use_cuda:
            data1[0] = data1[0].cuda()
            data1[1] = data1[1].cuda()

        out_1 = model(data1[0].float())[1].detach()
        sim_1 = torch.mm(out_1, normal_vec.t())

        label_list = torch.cat((label_list, data1[1].squeeze().cpu()))
        sim_1_list = torch.cat((sim_1_list, sim_1.squeeze().cpu()))
        if (batch + 1) % 100 == 0:
            print(f'Calculating Scores---- Evaluating: Batch {batch + 1} / {total_batch}')
    if score_folder is not None:
        np.save(os.path.join(score_folder, 'score.npy'), sim_1_list.numpy())
        print('score.npy is saved')
    return sim_1_list.numpy()


def get_new_represent(model, data_loader, use_cuda, batch_size, feature_dim):
    total_batch = int(len(data_loader))
    data_new = np.zeros([total_batch-1, batch_size, feature_dim])
    for batch, data in enumerate(data_loader):
        if use_cuda:
            data[0] = data[0].cuda()
        out = model(data[0].float())[1].detach().cpu().numpy()
        if batch < total_batch - 1:
            data_new[batch, :, :] = out

    data_new = data_new.reshape((total_batch-1)*batch_size, -1)
    print('Done! orginal data to the latent space')

    return data_new


def cal_score_downstream(model, normal_data, test_data, test_labels, downstream_model_type='IsoForest'):
    """
    Generate and save scores
    """
    test_len = len(test_data)
    test_labels = test_labels[:test_len]
    test_labels = test_labels * 2 - 1  # (1, 0) to (1, -1)
    # train
    print('train a downstream anomaly detector: {}'.format(downstream_model_type))
    if downstream_model_type is 'IsoForest':
        model = IsoForest(seed=1, train_data=normal_data, test_data=test_data, test_labels=test_labels,
                          n_estimators=100, max_samples=256,
                          contamination=0.1)

    # train IF model
    model.train()
    # predict
    model.predict()
    model.disp_results()

