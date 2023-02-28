from sklearn.svm import SVC 
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import random

import seaborn as sns
import warnings


def classifier(classifier_name, X_train, Y_train):
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if classifier_name == 'KNN':
        # Train KNeighborsClassifier Model
        KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
        KNN_Classifier.fit(X_train, Y_train)
        model = KNN_Classifier
    elif classifier_name == 'LGR':
        # Train LogisticRegression Model
        LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0, max_iter=500)
        LGR_Classifier.fit(X_train, Y_train)
        model = LGR_Classifier
    elif classifier_name == 'BNB':
        # Train Gaussian Naive Bayes Model
        BNB_Classifier = BernoulliNB()
        BNB_Classifier.fit(X_train, Y_train)
        model = BNB_Classifier
    elif classifier_name == 'DTC':
        # Train Decision Tree Model
        DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
        DTC_Classifier.fit(X_train, Y_train)
        model = DTC_Classifier
    elif classifier_name == 'SVM':
        SVC_Classifier = SVC(probability=False,  kernel="rbf")
        SVC_Classifier.fit(X_train, Y_train)
        model = SVC_Classifier
    elif classifier_name == 'MLP':
        MLP_Classifier = MLP(hidden_layer_sizes=(50,))
        MLP_Classifier.fit(X_train, Y_train)
        model = MLP_Classifier
    elif classifier_name == 'Consistency':
        consist_model = LabelSpreading(kernel='rbf', gamma=3)
        consist_model.fit(X_train, Y_train)
        model = consist_model
    else:
        print('ERROR: Unrecognized type of classifier')
    # evaluate(classifier_name, model, X_train, Y_train)
    return model


def evaluate(classifier_name, model, X, Y):
    scores = cross_val_score(model, X, Y, cv=5)
    Y_pre = model.predict(X)
    evaluate_sub(classifier_name, Y, Y_pre)
    print("Cross Validation Mean Score:" "\n", scores.mean())


def evaluate_only(classifier_name, model, X, Y):
    Y_pre = model.predict(X)
    evaluate_sub(classifier_name, Y, Y_pre)


def evaluate_sub(classifier_name, Y, Y_pre):
    accuracy = metrics.accuracy_score(Y, Y_pre)
    confusion_matrix = metrics.confusion_matrix(Y, Y_pre)
    classification = metrics.classification_report(Y, Y_pre)
    print()
    print('============================== {} Model Evaluation =============================='.format(classifier_name))
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()


def get_score(classifier_name, model, X_train):

    if classifier_name == 'KNN':
        # Train KNeighborsClassifier Model
        y_pred_score = model.predict_proba(X_train)[:, 1]
    elif classifier_name == 'LGR':
        # Train LogisticRegression Model
        y_pred_score = model.predict_proba(X_train)[:, 1]
    elif classifier_name == 'BNB':
        # Train Gaussian Naive Bayes Model
        y_pred_score = model.predict_proba(X_train)[:, 1]
    elif classifier_name == 'DTC':
        # Train Decision Tree Model
        y_pred_score = model.predict_proba(X_train)[:, 1]
    elif classifier_name == 'SVM':
        y_pred_score = model.decision_function(X_train)
    elif classifier_name == 'MLP':
        y_pred_score = model.predict_proba(X_train)[:, 1]
    else:
        print('ERROR: Unrecognized type of classifier')

    return y_pred_score


