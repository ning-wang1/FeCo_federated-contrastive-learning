import os
import copy
import torch
import numpy as np
from test import get_normal_vector, cal_score
from utils.utils import evaluate, get_score
from utils.utils import split_evaluate


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def test_inference(args, model, train_normal, test_data, test_labels, plot=False):
    model.eval()
    # Test inference after completion of training
    train_normal_loader_for_test = torch.utils.data.DataLoader(
        train_normal,
        batch_size=args.cal_vec_batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True,
    )
    normal_vec = get_normal_vector(model, train_normal_loader_for_test,
                                   args.cal_vec_batch_size,
                                   args.latent_dim,
                                   args.use_cuda)
    np.save(os.path.join(args.normvec_folder, 'normal_vec.npy'), normal_vec.cpu().numpy())
    cal_score(model, normal_vec, test_loader, args.score_folder, args.use_cuda)

    score = get_score(args.score_folder)
    best_acc, best_threshold, AUC = evaluate(score, test_labels, False, model_id='best')
    print(f'Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 6)} | AUC: {round(AUC, 4)}')
    split_evaluate(test_labels, score, plot=plot, filename='./result/detection/contrastive')

    return AUC
