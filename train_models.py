from lightgbm import LGBMClassifier
import re
import numpy as np
import pandas as pd
import gc
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from contextlib import contextmanager

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

global seed
seed = 1001


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified=False):
    # Divide in training/validation set
    train_df = df.copy()

    print("Starting LightGBM. Train shape: {}".format(train_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    # sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR']] # 'SK_ID_BUREAU', 'SK_ID_PREV', 'index' # deleted

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        print("Iteration : ", n_fold)
        # train_idx = index of rows from the training set
        # valid_idx = index of rows from the validation set
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        train_x = train_x.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        valid_x = valid_x.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        # train_y = train_y.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        print("Saving X_train, X_val, y_train and y_val")
        train_x.to_csv("dataset/cleaned/LGBM/X_train_fold_{}.csv".format(n_fold), index=False)
        valid_x.to_csv("dataset/cleaned/LGBM/X_val_fold_{}.csv".format(n_fold), index=False)

        # print("HERE", train_y.shape, valid_y.shape)
        train_y_df = pd.DataFrame(data=train_y, columns=["target"], index=train_x.index)
        valid_y_df = pd.DataFrame(data=valid_y, columns=["target"], index=valid_x.index)

        train_y_df.to_csv("dataset/cleaned/LGBM/y_train_fold_{}.csv".format(n_fold), index=False)
        valid_y_df.to_csv("dataset/cleaned/LGBM/y_val_fold_{}.csv".format(n_fold), index=False)

        print("Training LGBM")
        print("Training shape :", train_x.shape, "Validation shape :", valid_x.shape)
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        # save model
        # print("Saving LGBM")
        # filename_txt = 'models/LGBM/LGBMClassifier_fold_{}.txt'.format(n_fold)
        # clf.booster_.save_model(filename_txt)

        # save the model to disk
        print("Saving LGBM with joblib")
        filename_joblib = 'models/LGBM/LGBMClassifier_fold_{}.joblib'.format(n_fold)
        joblib.dump(clf, filename_joblib)

        # load model
        print("Loading LGBM")
        clf = joblib.load(filename_joblib)
        # clf = lgb.Booster(model_file='models/LGBMClassifier_fold_{}.txt'.format(i))
        # with Booster use predict : returns the proba

        print(
            "Predicting proba for class 1")  # [:, 1] to get the column for class 1 proba, else saving two columns => problem
        # valid_idx = validation set index
        # saving the prediction proba of validation set into oof_preds np array
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Plot feature importance
    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('models/LGBM/lgbm_importances01.png')


def modelling_lightgbm(df_path, debug=False):
    print("__Loading DataFrame__")
    df = pd.read_csv(df_path)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=10, stratified=False)


if __name__ == "__main__":
    with timer("Full model run"):
        print("Running time 2h30min lol : 9026s")
        modelling_lightgbm(df_path="dataset/cleaned/data_train_preprocessed_vf.csv")
