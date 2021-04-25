
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch

from test import split_acc_diff_threshold
from utils.utils import AverageMeter, l2_normalize
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.setup_NSL import NSL_data


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, idxs_normal, idxs_anormal, dataset_normal, dataset_anormal,
                 batch_logger, epoch_logger, memory_bank):
        self.args = args
        self.batch_logger = batch_logger
        self.epoch_logger = epoch_logger

        self.train_normal_loader, self.train_anormal_loader, self.valid_loader, self.test_loader = \
            self.train_val_test(list(idxs_normal), list(idxs_anormal), dataset_normal, dataset_anormal)

        self.device = 'cuda' if args.use_cuda else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

        len_neg = len(idxs_normal)
        len_pos = len(idxs_anormal)

        self.nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
        self.criterion = NCECriterion(len_neg)
        self.memory_bank = memory_bank

    def concat_dataset(self, ds1, ds2):
        concat_ds = ConcatDataset([ds1, ds2])
        data1 = concat_ds[0]
        labels1 = concat_ds[1]
        data2 = concat_ds[2]
        labels2 = concat_ds[3]
        data = np.concatenate([data1, data2], axis=0)
        labels = np.concatenate([labels1, labels2])
        ds_new = NSL_data(data, labels)
        return ds_new

    def train_val_test(self, idxs_normal, idxs_anormal, dataset_normal, dataset_anormal):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        normal_len = len(idxs_normal)
        anormal_len = len(idxs_anormal)

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_normal_train = idxs_normal[0:int(0.8 * normal_len)]
        idxs_anormal_train = idxs_anormal[:int(0.8 * anormal_len)]

        idxs_normal_val = idxs_normal[int(0.8 * normal_len):int(0.9 * normal_len)]
        idxs_anormal_val = idxs_anormal[int(0.8 * anormal_len):int(0.9 * anormal_len)]
        combined_val = self.concat_dataset(ds1=dataset_normal[idxs_normal_val],
                                           ds2=dataset_anormal[idxs_anormal_val])

        idxs_normal_test = idxs_normal[int(0.9 * normal_len):]
        idxs_anormal_test = idxs_anormal[int(0.9 * anormal_len):]
        combined_test = self.concat_dataset(ds1=dataset_normal[idxs_normal_test],
                                            ds2=dataset_anormal[idxs_anormal_test])

        train_normal_loader = DataLoader(DatasetSplit(dataset_normal, idxs_normal_train),
                                         batch_size=self.args.n_train_batch_size, shuffle=True)

        train_anormal_loader = DataLoader(DatasetSplit(dataset_anormal, idxs_anormal_train),
                                          batch_size=self.args.n_train_batch_size, shuffle=True)

        valid_loader = DataLoader(DatasetSplit(combined_val, np.arange(len(combined_val))),
                                  batch_size=self.args.val_batch_size, shuffle=False)

        test_loader = DataLoader(DatasetSplit(combined_test, np.arange(len(combined_test))),
                                 batch_size=self.args.val_batch_size, shuffle=False)

        return train_normal_loader, train_anormal_loader, valid_loader, test_loader

    def update_weights(self, model, model_head, epoch):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.learning_rate,
                                    momentum=self.args.momentum,
                                    dampening=self.args.dampening,
                                    weight_decay=self.args.weight_decay,
                                    nesterov=self.args.nesterov)
        losses = AverageMeter()
        prob_meter = AverageMeter()

        model.train()
        model_head.train()
        for batch, ((normal_data, idx_n), (anormal_data, idx_a)) in enumerate(
                zip(self.train_normal_loader, self.train_anormal_loader)):
            if normal_data.size(0) != self.args.n_train_batch_size:
                break
            data = torch.cat((normal_data, anormal_data), dim=0)  # n_vec as well as a_vec are all normalized value
            if self.args.use_cuda:
                data = data.cuda()
                idx_a = idx_a.cuda()
                idx_n = idx_n.cuda()
                normal_data = normal_data.cuda()

            # ================forward====================
            unnormed_vec, normed_vec = model(data.float())
            vec = model_head(unnormed_vec)
            n_vec = vec[0:self.args.n_train_batch_size]
            a_vec = vec[self.args.n_train_batch_size:]
            outs, probs = self.nce_average(n_vec, a_vec, idx_n, idx_a,
                                           normed_vec[0:self.args.n_train_batch_size])
            loss = self.criterion(outs)

            # ================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===========update memory bank===============
            model.eval()
            _, n = model(normal_data.float())
            n = n.detach()
            average = torch.mean(n, dim=0, keepdim=True)
            if len(self.memory_bank) < self.args.memory_bank_size:
                self.memory_bank.append(average)
            else:
                self.memory_bank.pop(0)
                self.memory_bank.append(average)
            model.train()

            # ===============update meters ===============
            losses.update(loss.item(), outs.size(0))
            prob_meter.update(probs.item(), outs.size(0))

            # =================logging=====================
            self.batch_logger.log({
                'epoch': epoch,
                'batch': batch,
                'loss': losses.val,
                'probs': prob_meter.val,
                'lr': optimizer.param_groups[0]['lr']
            })
            if batch % 100 == 0:
                print(f'Training Process is running: {epoch}  | Batch: {batch} '
                      f'| Loss: {losses.val} ({losses.avg})| Probs: {prob_meter.val} ({prob_meter.avg})')
        self.epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'probs': prob_meter.avg,
            'lr': optimizer.param_groups[0]['lr']
        })

        return model.state_dict(), model_head.state_dict(), self.memory_bank, losses.avg

    def inference(self, model):
        normal_vec = torch.mean(torch.cat(self.memory_bank, dim=0), dim=0, keepdim=True)
        normal_vec = l2_normalize(normal_vec)

        model.eval()

        accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
            model, normal_vec, self.valid_loader, self.args.use_cuda)
        print(f'validation | Accuracy: {np.round(accuracy, 4)} | Normal Acc: {np.round(acc_n, 4)}'
              f' | Anormal Acc: {np.round(acc_a, 4)} | Threshold: {np.round(best_threshold, 4)}')

        accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list = split_acc_diff_threshold(
            model, normal_vec, self.test_loader, self.args.use_cuda)
        print(f'Testing    | Accuracy: {np.round(accuracy, 4)} | Normal Acc: {np.round(acc_n, 4)}'
              f' | Anormal Acc: {np.round(acc_a, 4)} | Threshold: {np.round(best_threshold, 4)}')

        return accuracy
