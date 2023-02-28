import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
import logging
import time
import torch.optim as optim
import numpy as np
import pickle
import csv

from utils.logs import AD_Log
from utils.utils import split_evaluate, get_threshold

input_size = 115


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        print("Loading model...")
        with open(filename, 'rb') as f:
            self.encoder = pickle.load(f)

        print("Model loaded.")


class Autoencoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=input_size,
                out_features=128
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.ReLU(),

            # decoder layer 3
            nn.Linear(
                in_features=64,
                out_features=128
            ),
            nn.ReLU(),
            # Decoder output 4
            nn.Linear(
                in_features=128,
                out_features=input_size
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        print("Loading model...")
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

        print("Model loaded.")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=input_size,
                out_features=86
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=86,
                out_features=58
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=58,
                out_features=38
            ),
            nn.ReLU(),
            # layer 4
            nn.Linear(
                in_features=38,
                out_features=29
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.fc(x)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=29,
                out_features=38
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=38,
                out_features=58
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=58,
                out_features=86
            ),
            nn.ReLU(),
            # output
            nn.Linear(
                in_features=86,
                out_features=input_size
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class AETrainer(object):
    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.01, n_epochs: int = 800, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.loss_f = F.mse_loss

    def train(self, dataset, ae_net):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        _, train_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'adam')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        ae_net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs.float())
                loss = self.loss_f(outputs, inputs.float())
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        logger.info('Pretraining time: %.3f' % pretrain_time)
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset, ae_net, file_path):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, _, test_loader, valid_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        label_score = []
        ae_net.eval()

        valid_label_score = []
        with torch.no_grad():
            # calculate scores for the validation dataset
            for data in valid_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs.float())
                valid_scores = self.loss_f(outputs, inputs.float(), reduce=False)
                valid_scores = torch.mean(valid_scores, dim=1)
                valid_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                          valid_scores.cpu().data.numpy().tolist()))

            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                # outputs = ae_net(inputs.float())
                outputs = ae_net(inputs.float())
                scores = self.loss_f(outputs, inputs.float(), reduce=False)
                scores = torch.mean(scores, dim=1)
                loss = torch.mean(scores)

                # Save (label, score) in a list
                label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

        # set the threshold using the validation scores
        valid_labels, valid_scores = zip(*valid_label_score)
        idxs = np.where(np.array(valid_labels) == 0)
        normal_scores_valid = np.array(valid_scores)[idxs]
        th = get_threshold(normal_scores_valid, percent=99)
        dic = dict()
        split_evaluate(labels, scores, plot=True, filename=file_path + 'ae', perform_dict=dic)

        with open(file_path + 'ae.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=dic.keys())
            writer.writeheader()
            writer.writerow(dic)
        # split_evaluate(labels, scores, filename='./result/detection/ae')

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
































# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from utils.setup_BaIoT import BaIoT, BaIoT_data, DEVICE_NAMES
#
#
# def autoencoder():
#     input_shape=(784,)
#     model = Sequential()
#     model.add(Dense(64, activation='relu', input_shape=input_shape))
#     model.add(Dense(784, activation='sigmoid'))
#     return model
#
#
# def deep_autoencoder():
#     input_shape=(784,)
#     model = Sequential()
#     model.add(Dense(128, activation='relu', input_shape=input_shape))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(784, activation='sigmoid'))
#     return model
#
#
# def convolutional_autoencoder():
#
#     input_shape=(28,28,1)
#     n_channels = input_shape[-1]
#     model = Sequential()
#     model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
#     model.add(MaxPool2D(padding='same'))
#     model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
#     model.add(MaxPool2D(padding='same'))
#     model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
#     model.add(UpSampling2D())
#     model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
#     model.add(UpSampling2D())
#     model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
#     model.add(Conv2D(n_channels, (3,3), activation='sigmoid', padding='same'))
#     return model
#
#
# def load_model(name):
#     if name=='autoencoder':
#         return autoencoder()
#     elif name=='deep_autoencoder':
#         return deep_autoencoder()
#     elif name=='convolutional_autoencoder':
#         return convolutional_autoencoder()
#     else:
#         raise ValueError('Unknown model name %s was given' % name)
#
#
# class AETrainer(object):
#     def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
#                  batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
#         super().__init__()
#         self.optimizer_name = optimizer_name
#         self.lr = lr
#         self.n_epochs = n_epochs
#         self.lr_milestones = lr_milestones
#         self.batch_size = batch_size
#         self.weight_decay = weight_decay
#         self.device = device
#         self.loss = 'mean_squared_error'
#         self.n_jobs_dataloader = n_jobs_dataloader
#
#     def train_ae(self, dataset):
#
#         # prepare normal dataset
#         x_train, y_train = dataset.x_train, dataset.y_train
#         x_test, y_test = dataset.x_test, dataset.y_test
#
#         # instantiate model
#         model = load_model('deep_autoencoder')
#         # compile model
#         model.compile(optimizer=self.optimizer, loss=self.loss)
#
#         # train on only normal training data
#         model.fit(
#             x=x_train,
#             y=x_train,
#             epochs=self.n_epochs,
#             batch_size=self.batch_size,
#         )
#
#         # test
#         losses = []
#         for x in x_test:
#             # compule loss for each test sample
#             x = np.expand_dims(x, axis=0)
#             loss = model.test_on_batch(x, x)
#             losses.append(loss)
#
#         # plot
#         plt.plot(range(len(losses)), losses, linestyle='-', linewidth=1, label='deep ae')
#
#         # delete model for saving memory
#         del model
#
#
# if __name__ == '__main__':
#     AE = AETrainer()
#     data = BaIoT(DEVICE_NAMES[0])
#     AE.train_ae(data)
#
#
#
#
