import os
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import pickle as pickle
from utils.logs import AD_Log
from utils.utils import split_evaluate


class IsoForest(object):

    def __init__(self, seed, train_data, test_data, test_labels, n_estimators=100,
                 max_samples='auto', contamination=0.1, **kwargs):

        # initialize
        self.train_data = train_data
        self.test_data = test_data
        self.test_labels = test_labels
        self.isoForest = None
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.initialize_isoForest(seed=seed, **kwargs)

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
        self.test_time = 0

        # Scores and AUC
        self.diag = dict()
        self.diag['train'] = {}
        self.diag['test'] = {}

        self.diag['scores'] = np.zeros((len(self.test_labels), 1))
        self.diag['auc'] = np.zeros(1)
        self.diag['acc'] = np.zeros(1)

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_isoForest(self, seed=0, **kwargs):

        self.isoForest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                         contamination=self.contamination, n_jobs=-1, random_state=seed, **kwargs)

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self):

        if self.train_data.ndim > 2:
            X_train_shape = self.train_data.shape
            X_train = self.train_data.reshape(X_train_shape[0], -1)
        else:
            X_train = self.train_data

        print("Starting training...")
        self.start_clock()

        self.isoForest.fit(X_train.astype(np.float32))

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self):
        X = self.test_data
        y = self.test_labels

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], -1)

        print("Starting prediction...")
        self.start_clock()

        scores = self.isoForest.decision_function(X.astype(np.float32))  # compute anomaly score
        y_pred = self.isoForest.predict(X.astype(np.float32))  # get prediction

        self.diag['scores'][:, 0] = scores.flatten()
        self.diag['acc'][0] = 100.0 * sum(y == y_pred) / len(y)
        auc = roc_auc_score(y, scores.flatten())
        self.diag['auc'][0] = auc

        print("\n--------- result using API prediction -------")
        print('auc test', self.diag['auc'][-1])
        print('acc test', self.diag['acc'][-1])

        # show the accuracy on normal set and anormal set separately
        split_evaluate(y, scores.flatten(), filename='./result/detection/iso_forest')

        self.stop_clock()
        self.test_time = self.clocked

    def dump_model(self, filename=None):
        with open(filename, 'wb') as f:
            pickle.dump(self.isoForest, f)

        print("Model saved in %s" % filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        print("Loading model...")
        with open(filename, 'rb') as f:
            self.isoForest = pickle.load(f)

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


