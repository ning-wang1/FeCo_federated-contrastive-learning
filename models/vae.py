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

from utils.logs import AD_Log
from utils.utils import split_evaluate, get_threshold

input_size = 121
layer1_size = 512
layer2_size = 256
rep_dim = 32


class NSL_MLP_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_idx):
        x = batch['feature']
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)

    def dump_model(self, filename=None):
        with open(filename, 'wb') as f:
            pickle.dump(self.encoder, f)

        print("Model saved in %s" % filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        print("Loading model...")
        with open(filename, 'rb') as f:
            self.encoder = pickle.load(f)

        print("Model loaded.")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=input_size,
                out_features=512
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=1024
            ),
            nn.ReLU(),
        )

        # output
        self.mu = nn.Linear(
            in_features=1024,
            out_features=100
        )
        self.logvar = nn.Linear(
            in_features=1024,
            out_features=100
        )

    def forward(self, x):
        h = self.fc(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=100,
                out_features=1024
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=1024,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # output
            nn.Linear(
                in_features=512,
                out_features=input_size
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


class AETrainer(object):
    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
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

    def train(self, dataset, ae_net):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get train data loader
        train_loader, _, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

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
                # outputs = ae_net(inputs.float())
                outputs, mu, logvar = ae_net(inputs.float())
                # scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                # loss = torch.mean(scores)
                loss = ae_net.loss_function(outputs, inputs.float(), mu, logvar)/self.batch_size
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

    def test(self, dataset, ae_net):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        _, test_loader, valid_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

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
                outputs, _, _ = ae_net(inputs.float())
                valid_scores = -torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                valid_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                          valid_scores.cpu().data.numpy().tolist()))

            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                # outputs = ae_net(inputs.float())
                outputs, _, _ = ae_net(inputs.float())
                scores = -torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
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
        idxs = np.where(np.array(valid_labels) == 1)
        normal_scores_valid = np.array(valid_scores)[idxs]
        th = get_threshold(normal_scores_valid, percent=3)
        split_evaluate(labels, scores, plot=True, filename='./result/detection/ae', manual_th=th)
        # split_evaluate(labels, scores, filename='./result/detection/ae')

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')
