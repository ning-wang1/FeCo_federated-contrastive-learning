import torch
import torch.nn as nn
from utils.utils import l2_normalize
import numpy as np
import os
from models.iso_forest import IsoForest
from models.mlp import MLP
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from detection_two_classes import get_acc, early_stop, mlp_predict
from utils.setup_NSL import NSL_KDD, NSL_data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelSpreading

from utils.utils import l2_normalize, get_score, get_threshold


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
        if batch < total_batch - 1:
            out = model(data[0].float())[1].detach().cpu().numpy()
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


def train_ids(input_shape, trainloader, validationloader, testloader, output_shape, save_folder=None):
    """
    Standard neural network training procedure.
    """
    max_epochs = 100
    moving_average = 100

    mlp = MLP(input_shape, 256, 256, output_size=output_shape)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(mlp.parameters(), lr=0.01, weight_decay=0.001)

    mlp_list = []
    crt_list = []

    for epoch in range(0, max_epochs):

        current_loss = 0
        for i, data in enumerate(trainloader, 0):
            input, target = data
            input, target = Variable(input), Variable(target)

            mlp.zero_grad()
            output, _ = mlp(input.float())
            loss = criterion(output, target.long())

            loss.backward()
            optimizer.step()

            loss = loss.item()
            current_loss += loss

        train_c = get_acc(mlp, trainloader)
        test_c = get_acc(mlp, testloader)
        valid_c = get_acc(mlp, validationloader)
        print('[ %d ]Training loss : %.3f' % (epoch + 1, current_loss / len(trainloader)))
        print('[ %d ] Training accuracy: %.4f, Validation Acc: %.4f, Testing Acc: %.4f' %
              (epoch + 1, train_c, valid_c, test_c))

        mlp_list.append(mlp)
        crt_list.append(valid_c)

        if epoch >= moving_average:
            if early_stop(crt_list, moving_average):
                print('Early stopping.')
                index = int(len(mlp_list) - moving_average / 2)
                if save_folder is not None:
                    torch.save(mlp.state_dict(), os.path.join(save_folder, 'nn'))
                return mlp_list[index]

    return mlp


def train_downstream_classifier(args, model, x_train, y_train, x_test, y_test):
    """
    Generate and save scores
    """
    test_data = NSL_data(x_test, y_test)
    test_loader_0 = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    train_data = NSL_data(x_train, y_train)
    train_loader_0 = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    test_out = get_new_represent(model, test_loader_0, args.use_cuda, args.val_batch_size, args.latent_dim)
    train_out = get_new_represent(model, train_loader_0, args.use_cuda, args.val_batch_size, args.latent_dim)

    min_max_scaler = MinMaxScaler()
    train_out = min_max_scaler.fit_transform(train_out)
    test_out = min_max_scaler.transform(test_out)

    train_length = train_out.shape[0]
    train_out_new = train_out[0: int(0.8*train_length)]
    valid_out = train_out[int(0.8 * train_length):, :]

    train_new = NSL_data(train_out_new, y_train[0: int(0.8*train_length)])
    valid_new = NSL_data(valid_out, y_train[int(0.8 * train_length):])
    test_new = NSL_data(test_out, y_test)

    train_loader = torch.utils.data.DataLoader(train_new, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_new, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_new, batch_size=64, shuffle=False)
    # train
    print('train a downstream anomaly detector')

    mlp_model = train_ids(args.latent_dim, train_loader, valid_loader, test_loader, output_shape=2)
    # mlp_model = train_ids(110, train_loader_0, test_loader_0, test_loader_0, output_shape=2)
    y_pred = mlp_predict(mlp_model, test_loader)
    y_pred = np.argmax(y_pred, axis=1)
    # present_split_acc(y_pred, data.test_labels)
    # split_evaluate(data.test_labels, y_pred[:, 1], plot=True, filename=args.plot_folder)


def multiclass_test(args, all_data, model):
    attacks = {'Normal': 1.0, 'DoS': 0.0, 'Probe': 2.0, 'R2L': 3.0, 'U2R': 4.0}

    dict_th = {}
    dict_vec = {}
    dict_attack = {i: np.array([], dtype='int64') for i in attacks.keys()}
    train_data = all_data.train_data
    valid_data = all_data.validation_data
    test_data = all_data.test_data
    train_labels = all_data.y_train_multi_class.astype('long')
    valid_labels = all_data.y_valid_multi_class.astype('long')
    test_labels = all_data.y_test_multi_class.astype('long')

    for name, attack_label in attacks.items():
        idx = np.where(train_labels == attack_label)
        data = train_data[idx]
        dict_attack[name] = data

        dataset = NSL_data(data, train_labels[idx])
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.cal_vec_batch_size,
                                                  shuffle=True, num_workers=args.n_threads, pin_memory=True)
        dict_vec[name] = get_normal_vector(model, data_loader, args.cal_vec_batch_size, args.latent_dim, args.use_cuda)

        # for valid data
        idx = np.where(valid_labels == attack_label)
        data = valid_data[idx]
        dataset = NSL_data(data, valid_labels[idx])
        valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )

        # compute a decision-making threshold using the validation dataset
        valid_scores = cal_score(model, dict_vec[name], valid_loader, None, args.use_cuda)
        dict_th[name] = get_threshold(valid_scores, percent=5)
        print(f'the threshold for class {name} is set as {dict_th[name]}')

    # evaluating the scores of the test dataset and show the IDS performance
    dataset = NSL_data(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    attack_name = ['DoS', 'Normal', 'Probe', 'R2L', 'U2R']
    score = np.zeros([len(test_labels), 5])
    for i in range(5):
        score[:, i] = cal_score(model, dict_vec[attack_name[i]], test_loader, use_cuda=args.use_cuda)
    y = np.argmax(score, axis=1)
    print(np.sum(y == test_labels))

    # split_evaluate(all_data.test_labels, score, plot=True,
    #                filename=f'{args.result_folder}contrastive', manual_th=th, perform_dict=performance_dict)
    # per_class_acc(all_data.y_test_multi_class, score, th, perform_dict=performance_dict)


def get_vectors(model, data_loader, data_len, batch_size, latent_dim, use_cuda):
    total_batch = int(len(data_loader))
    print("------------------- Collecting Vectors --------------------")
    if use_cuda:
        normal_vec = torch.zeros((data_len, latent_dim)).cuda()
    else:
        normal_vec = torch.zeros((data_len, latent_dim))
    for batch, (normal_data, idx) in enumerate(data_loader):
        if use_cuda:
            normal_data = normal_data.cuda()
        _, outputs = model(normal_data.float())
        outputs = outputs.detach()

        if batch < total_batch - 1:
            normal_vec[batch * batch_size: (batch + 1) * batch_size] = outputs
            if (batch + 1) % 1000 == 0:
                print(f'Calculating Normal Vector: Batch {batch + 1} / {total_batch}')
        else:
            normal_vec[batch * batch_size: ] = outputs

    return normal_vec


def manifold(train_data, train_labels, test_data, test_labels):
    print('training consist model')
    consist_model = LabelSpreading(gamma=3)
    consist_model.fit(train_data, train_labels)
    pred_consist = consist_model.predict_proba(test_data)  # the output of consistency model

    return pred_consist


def detect_with_manifold(args, model, train_normal, train_anormal, test_dataloader,
                         normal_len, anormal_len, test_len, test_labels):
    batch_size = args.cal_vec_batch_size
    normal_data = get_vectors(model, train_normal, normal_len, batch_size, args.latent_dim, args.use_cuda)
    anormal_data = get_vectors(model, train_anormal, anormal_len, batch_size, args.latent_dim, args.use_cuda)
    test_data = get_vectors(model, test_dataloader, test_len, args.val_batch_size, args.latent_dim, args.use_cuda)

    all_data = np.concatenate((normal_data.cpu(), anormal_data.cpu()), axis=0)
    all_labels = np.concatenate((np.ones(normal_data.shape[0]), np.zeros(anormal_data.shape[0])))

    idx = np.arange(all_data.shape[0])
    np.random.shuffle(idx)
    selected_idx = idx[0: 20000]

    pred_consist = manifold(all_data[selected_idx], all_labels[selected_idx], test_data.cpu(), test_labels)
    return pred_consist



