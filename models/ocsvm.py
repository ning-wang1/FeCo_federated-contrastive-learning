import time
import os
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from utils.logs import AD_Log
from matplotlib import pyplot as plt
import pickle
from utils.utils import split_evaluate


def weighted_degree_kernel(X1, X2, degree=1, weights=1):
    """
    Compute the weighted degree kernel matrix for one-hot encoded sequences.
    X1, X2: Tensor of training examples with shape
        (n_examples, 1, len_dictionary, len_sequence)
    degree: Degree of kernel
    weights: list or tuple with degree weights
    :return: Kernel matrix K
    """

    # assert degree == len(weights)
    assert degree == weights
    Klist = [None] * degree

    na = np.newaxis
    ones = np.ones(X2.shape)
    K = np.logical_and((X1[:, na, :, :, :] == X2[na, :, :, :, :]),
                       (X1[:, na, :, :, :] == ones[na, :, :, :, :]))
    K = K.sum(axis=(2, 3))

    # compute kernel matrix for each degree
    Klist[0] = K.sum(axis=2)
    for i in range(degree-1):
        K = (K[:, :, :-1] == K[:, :, 1:])
        Klist[i+1] = K.sum(axis=2)

    # compute weighted degree kernel
    K = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(degree):
        K += weights[i] * Klist[i]

    return K


def degree_kernel(X1, X2, degree=1):
    """
    Compute the degree kernel matrix for one-hot encoded sequences.
    X1, X2: Tensor of training examples with shape
        (n_examples, 1, len_dictionary, len_sequence)
    degree: Degree of kernel
    :return: Kernel matrix K
    """

    na = np.newaxis
    ones = np.ones(X2.shape)

    K = np.logical_and((X1[:, na, :, :, :] == X2[na, :, :, :, :]),
                       (X1[:, na, :, :, :] == ones[na, :, :, :, :]))

    K = K.sum(axis=(2, 3))

    for i in range(degree-1):
        K = (K[:, :, :-1] == K[:, :, 1:])

    K = K.sum(axis=2)

    return K


class SVM(object):

    def __init__(self, loss, normal_data, data, kernel, **kwargs):

        # config
        self.svm_C = 0.3
        self.svm_nu = 0.5

        # initialize
        self.svm = None
        self.cv_svm = None
        self.loss = loss
        self.kernel = kernel
        self.K_train = None
        self.K_val = None
        self.K_test = None
        self.K_train_normal = None
        self.nu = None
        self.gamma = None
        self.initialize_svm(loss, **kwargs)
        self.normal_data = normal_data
        self.data = data

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
        self.val_time = 0
        self.test_time = 0

        # Scores and AUC
        self.diag = dict()

        self.diag['train'] = {}
        self.diag['val'] = {}
        self.diag['test'] = {}

        self.diag['train']['scores'] = np.zeros((len(self.data.train_labels), 1))
        self.diag['val']['scores'] = np.zeros((len(self.data.validation_labels), 1))
        self.diag['test']['scores'] = np.zeros((len(self.data.test_labels), 1))

        self.diag['train']['auc'] = np.zeros(1)
        self.diag['val']['auc'] = np.zeros(1)
        self.diag['test']['auc'] = np.zeros(1)

        self.diag['train']['acc'] = np.zeros(1)
        self.diag['val']['acc'] = np.zeros(1)
        self.diag['test']['acc'] = np.zeros(1)

        self.rho = None

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_svm(self, loss, **kwargs):

        assert loss in ('SVC', 'OneClassSVM')

        if self.kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
            kernel = self.kernel
        else:
            kernel = 'precomputed'

        if loss == 'SVC':
            self.svm = svm.SVC(kernel=kernel, C=self.svm_C, **kwargs)
        if loss == 'OneClassSVM':
            self.svm = svm.OneClassSVM(kernel=kernel, nu=self.svm_nu, **kwargs)
            self.cv_svm = svm.OneClassSVM(kernel=kernel, nu=self.svm_nu, **kwargs)

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def train(self, GridSearch=True, **kwargs):

        if self.data.train_data.ndim > 2:
            X_train_shape = self.data.train_data.shape
            X_train = self.data.train_data.reshape(X_train_shape[0], np.prod(X_train_shape[1:]))
        else:
            X_train = self.data.train_data

        print("Starting training...")
        self.start_clock()

        if self.loss == 'SVC':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set='train', **kwargs)
                self.svm.fit(self.K_train, self.data.train_labels)
            else:
                self.svm.fit(X_train, self.data.train_labels)

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set='train', **kwargs)
                self.svm.fit(self.K_train_normal)
            else:

                if GridSearch and self.kernel == 'rbf':

                    # use grid search cross-validation to select gamma
                    print("Using GridSearchCV for hyperparameter selection...")
                    self.data.n_val = len(self.data.validation_labels)
                    self.data.n_test = len(self.data.test_labels)

                    self.diag['val']['scores'] = np.zeros((len(self.data.validation_labels), 1))
                    self.diag['test']['scores'] = np.zeros((len(self.data.test_labels), 1))

                    cv_auc = 0.0
                    cv_acc = 0

                    for gamma in np.logspace(-10, -1, num=10, base=2):

                        # train on selected gamma
                        self.cv_svm = svm.OneClassSVM(kernel='rbf', nu=self.svm_nu, gamma=gamma)
                        self.cv_svm.fit(self.normal_data)

                        # predict on small hold-out set
                        self.predict(which_set='val')

                        # save model if AUC on hold-out set improved
                        if self.diag['val']['auc'] > cv_auc:
                            self.svm = self.cv_svm
                            self.nu = self.svm_nu
                            self.gamma = gamma
                            cv_auc = self.diag['val']['auc']
                            cv_acc = self.diag['val']['acc']

                    # save results of best cv run
                    self.diag['val']['auc'] = cv_auc
                    self.diag['val']['acc'] = cv_acc

                else:
                    # if rbf-kernel, re-initialize svm with gamma minimizing the
                    # numerical error
                    if self.kernel == 'rbf':
                        idxes = np.arange(len(self.normal_data))
                        np.random.shuffle(idxes)
                        sel_idxes = idxes[:300]
                        gamma = 1 / (np.max(pairwise_distances(self.normal_data[sel_idxes])) ** 2)
                        # self.svm = svm.OneClassSVM(kernel='rbf', nu=self.svm_nu, gamma=gamma)
                        print(f'>>>>>>>>>>>>>>>>>>>>......  svm_nu: {self.svm_nu} ')
                        self.svm = svm.OneClassSVM(kernel='rbf', nu=self.svm_nu, gamma='auto')

                    self.svm.fit(self.normal_data[idxes[:10000]])

                    self.nu = self.svm_nu
                    self.gamma = gamma

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self, save_path, which_set='train', **kwargs):

        assert which_set in ('train', 'val', 'test')

        if which_set == 'train':
            X = self.data.train_data
            y = self.data.train_labels * 2 - 1
        if which_set == 'val':
            X = self.data.validation_data
            y = self.data.validation_labels * 2 - 1
        if which_set == 'test':
            X = self.data.test_data
            y = self.data.test_labels * 2 - 1

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

        print("Starting prediction...")
        self.start_clock()

        if self.loss == 'SVC':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set=which_set, **kwargs)
                if which_set == 'train':
                    scores = self.svm.decision_function(self.K_train)
                if which_set == 'test':
                    scores = self.svm.decision_function(self.K_test)
            else:
                scores = self.svm.decision_function(X)
                y_pred = self.svm.predict(X)

            auc = roc_auc_score(y, scores)
            self.diag[which_set]['scores'] = scores
            self.diag[which_set]['auc'][0] = auc
            self.diag[which_set]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)

        if self.loss == 'OneClassSVM':

            if self.kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
                self.get_kernel_matrix(kernel=self.kernel, which_set=which_set, **kwargs)
                if which_set == 'train':
                    scores = self.svm.decision_function(self.K_train)
                    y_pred = self.svm.predict(self.K_train)
                if which_set == 'test':
                    scores = self.svm.decision_function(self.K_test)
                    y_pred = self.svm.predict(self.K_test)
            else:
                if which_set == "val":
                    scores = self.cv_svm.decision_function(X)
                    y_pred = self.cv_svm.predict(X)
                else:
                    scores = self.svm.decision_function(X)
                    y_pred = self.svm.predict(X)

            self.diag[which_set]['scores'][:, 0] = scores.flatten()
            self.diag[which_set]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)
            auc = roc_auc_score(y, scores.flatten())
            self.diag[which_set]['auc'][0] = auc

        print("\n--------- result using API prediction -------")
        print('auc test', self.diag[which_set]['auc'][-1])
        print('acc test', self.diag[which_set]['acc'][-1])

        # show the accuracy on normal set and anormal set separately
        split_evaluate(y, scores.flatten(), plot=True, filename=save_path + self.loss)

        self.stop_clock()
        if which_set == 'test':
            self.rho = -self.svm.intercept_[0]
            self.test_time = self.clocked
        if which_set == 'val':
            self.val_time = self.clocked

    def dump_model(self, filename=None):

        with open(filename, 'wb') as f:
            pickle.dump(self.svm, f)
        print("Model saved in %s" % filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        with open(filename, 'rb') as f:
            self.svm = pickle.load(f)

        print("Model loaded.")

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['train_auc'] = self.diag['train']['auc'][-1]
        self.ad_log['train_accuracy'] = self.diag['train']['acc'][-1]
        self.ad_log['train_time'] = self.train_time

        self.ad_log['val_auc'] = self.diag['val']['auc'][-1]
        self.ad_log['val_accuracy'] = self.diag['val']['acc'][-1]
        self.ad_log['val_time'] = self.val_time

        self.ad_log['test_auc'] = self.diag['test']['auc'][-1]
        self.ad_log['test_accuracy'] = self.diag['test']['acc'][-1]
        self.ad_log['test_time'] = self.test_time

        print('auc test', self.diag['test']['auc'][-1])
        print('acc test', self.diag['test']['acc'][-1])

        self.ad_log.save_to_file(filename=filename)

    def get_kernel_matrix(self, kernel, which_set='train', **kwargs):

        assert kernel in ('DegreeKernel', 'WeightedDegreeKernel')

        if kernel == 'DegreeKernel':
            kernel_function = degree_kernel
        if kernel == 'WeightedDegreeKernel':
            kernel_function = weighted_degree_kernel

        if which_set == 'train':
            self.K_train = kernel_function(self.data.train_data, self.data.train_data, **kwargs)
            self.K_train_normal = kernel_function(self.normal_data, self.normal_data, **kwargs)
        if which_set == 'val':
            self.K_val = kernel_function(self.data.validation_data, self.data.train_data, **kwargs)
        if which_set == 'test':
            self.K_test = kernel_function(self.data.test_data, self.data.train_data, **kwargs)
