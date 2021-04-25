import os
import time
import pickle
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import pairwise_distances

from utils.logs import AD_Log
from utils.utils import split_evaluate


class KDE(object):

    def __init__(self, train_data, test_data, test_labels, kernel, **kwargs):

        # initialize
        self.train_data = train_data
        self.test_data = test_data
        self.test_labels = test_labels
        self.kde = None
        self.kernel = kernel
        self.bandwidth = None
        self.initialize_kde(**kwargs)

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
        self.test_time = 0

        # Scores and AUC
        self.diag = dict()

        self.diag['train'] = {}
        self.diag['val'] = {}
        self.diag['test'] = {}

        self.diag['scores'] = np.zeros((len(self.test_labels), 1))
        self.diag['auc'] = np.zeros(1)
        self.diag['acc'] = np.zeros(1)

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_kde(self, **kwargs):

        self.kde = KernelDensity(kernel=self.kernel, **kwargs)
        self.bandwidth = self.kde.bandwidth

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self, bandwidth_GridSearchCV=True):

        if self.train_data.ndim > 2:
            X_train_shape = self.train_data.shape
            X_train = self.train_data.reshape(X_train_shape[0], -1)
        else:
            X_train = self.train_data

        print("Starting training...")
        self.start_clock()

        if bandwidth_GridSearchCV:
            # use grid search cross-validation to select bandwidth
            print("Using GridSearchCV for bandwidth selection...")

            # params = {'bandwidth': np.logspace(0.5, 5, num=10, base=2)}
            params = {'bandwidth': np.logspace(- 4.5, 5, num=20, base=2)}

            hyper_kde = GridSearchCV(KernelDensity(kernel=self.kernel), params, n_jobs=-1, cv=5, verbose=0)
            hyper_kde.fit(X_train)

            self.bandwidth = hyper_kde.best_estimator_.bandwidth
            self.kde = hyper_kde.best_estimator_
        else:
            # if exponential kernel, re-initialize kde with bandwidth minimizing
            # the numerical error
            if self.kernel == 'exponential':
                bandwidth = np.max(pairwise_distances(X_train)) ** 2
                self.kde = KernelDensity(kernel=self.kernel,
                                         bandwidth=bandwidth)

            self.kde.fit(X_train)

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self):
        X = self.test_data
        y = ((self.test_labels + 1) / 2).astype(int)

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], -1)

        print("Starting prediction...")
        self.start_clock()

        # evaluate by batches
        batch_size = 1000
        batch_num = int(np.floor(len(y)/batch_size))
        # batch_num=5

        # get the log-likelihood
        for i in range(batch_num):
            print("predicting test data: {}---{}".format(i*batch_size, (i+1)*batch_size))
            scores = self.kde.score_samples(X[i*batch_size: (i+1)*batch_size])  # log-likelihood
            self.diag['scores'][i*batch_size: (i+1)*batch_size, 0] = scores.flatten()

        scores = self.kde.score_samples(X[(i + 1) * batch_size:])  # log-likelihood
        self.diag['scores'][(i + 1) * batch_size:, 0] = scores.flatten()

        # show the accuracy on normal set and anormal set separately
        split_evaluate(y, self.diag['scores'].flatten(), filename='./result/detection/kde')

        self.stop_clock()
        self.test_time = self.clocked

    def dump_model(self, filename=None):

        with open(filename, 'wb') as f:
            pickle.dump(self.kde, f)
        print("Model saved in %s" % filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        with open(filename, 'rb') as f:
            self.kde = pickle.load(f)

        print("Model loaded.")

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['test_auc'] = self.diag['auc'][-1]
        self.ad_log['test_accuracy'] = self.diag['acc'][-1]
        self.ad_log['test_time'] = self.test_time

        print('auc test', self.diag['auc'][-1])
        print('acc test', self.diag['acc'][-1])

        self.ad_log.save_to_file(filename=filename)

    def disp_results(self):
        print('auc test', self.diag['auc'][-1])
        print('acc test', self.diag['acc'][-1])
