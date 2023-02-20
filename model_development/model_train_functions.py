from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import Pipeline as imbpipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

from tqdm import tqdm
import pickle
import joblib

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (recall_score, roc_auc_score, f1_score, roc_curve, auc, roc_curve,
                             precision_recall_curve, make_scorer, accuracy_score,
                             precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             fbeta_score, make_scorer)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, GridSearchCV

from time import time


def fit_and_predict(model, model_name, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    # save the model to disk
    print("Saving model with joblib")
    filename_joblib = 'models/{}.joblib'.format(model_name)
    joblib.dump(model, filename_joblib)
    return model.predict(X_test)


def f1_(y_test, predictions):
    return f1_score(y_test, predictions)


def recall_(y_test, predictions):
    return recall_score(y_test, predictions)


def auc_(y_test, predictions):
    return roc_auc_score(y_test, predictions)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    plt.show()


# not used
def make_grisearch_smote(model,
                         X,
                         y,
                         param_grid: dict,
                         k_fold=2):
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    stratified_kfold = StratifiedKFold(
        n_splits=k_fold,
        shuffle=True
    )

    pipeline = imbpipeline(steps=[['smote', SMOTE(random_state=11)],
                                  ['classifier', model]
                                  ])

    stratified_kfold = StratifiedKFold(n_splits=3,
                                       shuffle=True,
                                       random_state=11)

    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring=ftwo_scorer,
                               cv=stratified_kfold,
                               n_jobs=-1)

    grid_search.fit(X, y)

    return grid_search.best_score_, grid_search.best_params_


def make_grisearch(model,
                   X,
                   y,
                   param_grid: dict,
                   k_fold=3):
    stratified_kfold = StratifiedKFold(
        n_splits=k_fold,
        shuffle=True
    )

    ftwo_scorer = make_scorer(fbeta_score, beta=2)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=ftwo_scorer,
        cv=stratified_kfold,
        n_jobs=-1)
    print("Fitting Grid")
    t0 = time()
    grid_search.fit(X, y)
    t1 = time() - t0
    print("Computing time :", t1)
    return grid_search.best_score_, grid_search.best_params_


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]


def evaluate(model,
             model_name: str,
             X,
             y,
             k_fold=5,
             beta=3,
             show_confusion_matrice=False,
             # apply_undersampling=False,
             apply_smote=False,
             smote_params=None):  # ,
    # move_treshold=False,
    # treshold=None) -> tuple:
    '''
    take a model, X, y, apply smote or not, smote_params={'k_neighbors':int, 'sampling_strategy':float}
    and return 4 metrics with a n Kfold cros validated
    and "mean" confusion matrice (with possibility to display
    '''
    print("Starting time")
    t0 = time()
    # instanciate skf
    skf = StratifiedKFold(n_splits=k_fold, random_state=123, shuffle=True)

    scores = {'f_beta_score_': [],
              'precision_': [],
              'recall_': [],
              'accuracy_': [], }
    i = 0
    for train_ix, test_ix in skf.split(X, y):
        print("Iteration : ", i)
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        if apply_smote:
            oversample = SMOTE(k_neighbors=smote_params['k_neighbors'],
                               sampling_strategy=smote_params['sampling_strategy'])
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        # if apply_undersampling:
        #   undersample = NearMiss(version=1, n_neighbors=100, sampling_strategy=0.5)
        #  X_train, y_train = undersample.fit_resample(X_train, y_train)

        #  if move_treshold:
        #     model.fit(X_train, y_train)
        #    probabilities = model.predict_proba(X_test)[:, 1]
        #   predictions = adjusted_classes(probabilities, treshold)

        else:
            # predict
            print("Starting fit and predict")
            # Create local directories to save data
            os.makedirs("models/{}".format(model_name), exist_ok=True)  # just in case the folder doesn't exist
            model_filename = "{}/{}_fold_{}".format(model_name, model_name, i)
            predictions = fit_and_predict(model, model_filename, X_train, y_train, X_test)

        scores['f_beta_score_'].append(fbeta_score(y_test, predictions, beta=beta))
        scores['precision_'].append(accuracy_score(y_test, predictions))
        scores['recall_'].append(recall_score(y_test, predictions))
        scores['accuracy_'].append(accuracy_score(y_test, predictions))

        i += 1

    # return mean for metrics
    for metric, scores_values in scores.items():
        scores[metric] = np.mean(scores_values)

    if show_confusion_matrice:
        ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

    t1 = time() - t0
    print("Computing time :", t1)
    return pd.DataFrame.from_dict(scores, orient='index').rename(columns={0: model_name})


def get_feature_importance_model(model, columns):
    """

    :param model:
    :param columns:
    :return:
    :rtype:

    """
    global_feature_importance = model.feature_importances_.tolist()  # need this to serialize into json object

    # create dict {columns_name:model_feature_importance}
    dict_feature_importance = dict(zip(columns, global_feature_importance))  # model.feature_importances_
    # sorted by feature_importance
    dict_feature_importance = {k: v for k, v in
                               sorted(dict_feature_importance.items(), key=lambda item: item[1], reverse=True)}
    return dict_feature_importance


def plot_feature_importance(dict_feature_importance):
    """

    :param dict_feature_importance:
    :return: None
    :rtype: None
    """
    plt.rcParams['figure.autolayout'] = True
    plt.figure(figsize=(15, 8))
    sns.barplot(x=list(dict_feature_importance.values()), y=list(dict_feature_importance.keys()), orient='h',
                color='#BCD6AD')
    plt.show()
