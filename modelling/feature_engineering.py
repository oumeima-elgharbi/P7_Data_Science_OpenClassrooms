"""

Source : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script

Author : Oumeima EL GHARBI
Modified code : OneHotEncoder, installation, train/test
"""

# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables.
# Little feature selection is done and over-fitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
# Garbage collector
import gc
from time import time
from contextlib import contextmanager

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import pickle

import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Create local directories to save data
print("__Creating folders locally__")
os.makedirs("dataset/source", exist_ok=True)
os.makedirs("dataset/cleaned", exist_ok=True)
os.makedirs("models/preprocessing", exist_ok=True)

@contextmanager
def timer(title):
    """
    Computes processing time when called

    :param title: (string) Name of the task being timed
    :return: None
    :rtype: None
    """
    t0 = time()
    yield  # ??
    print("{} - done in {:.0f}s".format(title, time() - t0))


def OLD_one_hot_encoder(df, nan_as_category=True):  # TODO Delete
    """
    One-hot encoding for categorical columns with get_dummies

    :param df: (DataFrame)
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :return: The encoded dataframe and the list of the new columns generated
    :rtype: tuple
    """
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def one_hot_encoder(df, df_name, training=True, nan_as_category=True):
    """
    Training : if True we fit and save a OHE else we load it locally
    nan_as_category : if True then we keep the encoded columns generated by the OHE for NaN values, Else we delete these
    columns

    :param df: (DataFrame)
    :param df_name:
    :param training:
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :return: The encoded dataframe and the list of the new columns generated
    :rtype: tuple
    """
    original_columns = list(df.columns)

    filename_joblib = 'models/preprocessing/OneHotEncoder_{}.joblib'.format(df_name)
    filename_pickle = 'models/preprocessing/OneHotEncoder_with_columns_{}.sav'.format(df_name)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']  ## TODO
    print("Number of columns to encode :", len(categorical_columns))

    col_not_cat = [col for col in df.columns if df[col].dtype != 'object']  # or not in categorical_columns
    print("Number of columns to merge with the encoded df :", len(col_not_cat))

    if training:
        # 1) Creating instance of one-hot-encoder and Fit the encoder on the training set
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(df[categorical_columns])
        # encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=categorical_columns).fit(df)

        # if sparse=True (by default), we need to add .toarray() to encoded_categorical_data

        # encoder = ColumnTransformer(
        #   [('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns)],
        #   remainder='passthrough'
        # ).fit(df)  # handle_unknown='ignore', sparse_output=False

        # 2) save the OHE to disk
        print("Saving One Hot Encoder and categorical columns")

        model = {'encoder': encoder, 'encoder_features': categorical_columns}
        with open(filename_pickle, "wb") as f:
            pickle.dump(model, f)

        joblib.dump(encoder, filename_joblib)

    else:
        # load model
        print("Loading One Hot Encoder and columns")
        # encoder = joblib.load(filename_joblib)

        with open(filename_pickle, 'rb') as pickle_file:
            encoder_model = pickle.load(pickle_file)
        encoder = encoder_model["encoder"]
        categorical_columns = encoder_model["encoder_features"]

    encoded_categorical_data = encoder.transform(df[categorical_columns])

    # 3) we make a list of the columns names
    encoded_categorical_data_names = encoder.get_feature_names_out().tolist()
    print("We have indeed :", len(encoded_categorical_data_names),
          "labels after encoding the categorical variables with nan counted.")

    # 4) we recreate a dataframe with the column names and the numpy array
    df_encoded = pd.DataFrame(columns=encoded_categorical_data_names,
                              data=encoded_categorical_data,
                              index=df.index)

    # 5) Concatenate the two dataframes for the training set

    # when merging, we put the categorical features first so that the targets will be at the end of the dataframe.  // INVERSE
    df_all_encoded = pd.merge(df[col_not_cat].sort_index(), df_encoded.sort_index(), left_index=True, right_index=True)

    # nan_columns = [s for s in encoded_categorical_data_names if s.endswith("nan")]
    # print("HEREEE NAN", nan_columns)
    # The OHE will automatically create a nan column for NaN / if we don't want it, we drop these columns
    if not nan_as_category:
        # we remove NaN columns
        nan_columns = [s for s in encoded_categorical_data_names if s.endswith("nan")]
        print("We drop these columns :", nan_columns)
        df_all_encoded = df_all_encoded.drop(columns=nan_columns)

    new_columns = [c for c in df_all_encoded.columns if c not in original_columns]
    print("Number of new columns :", len(new_columns))

    ###
    # print(encoded_categorical_data_names)
    # l = [col for col in encoded_categorical_data_names if col not in new_columns]
    # print("HHEREE", l)
    return df_all_encoded, new_columns


def application(input_path, application_filename, num_rows=None, nan_as_category=True,
                training=True):  # TODO put back True
    """
    Preprocess application_train.csv (or application_test.csv)

    :param input_path: (string) the path to the file "application_train.csv"
    :param application_filename: (string) name of csv file
    :param num_rows: (int or None) if we do not want to read all the csv file but only a certain number of rows
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :param training:
    :return:
    :rtype: DataFrame
    """
    # Read data and merge
    df = pd.read_csv(input_path + application_filename, nrows=num_rows)
    print("Application samples: {}".format(len(df)))
    print("Application {} df shape:".format(application_filename), df.shape)

    # test_df = pd.read_csv(input_path + 'application_test.csv', nrows=num_rows)
    # print("Test samples: {}".format(len(test_df)))
    # print("Application test df shape:", test_df.shape)
    # df = df.append(test_df).reset_index()
    # print("Application train and test df shape:", df.shape)

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    # df1, cat_cols1 = one_hot_encoder(df, nan_as_category)
    df, cat_cols = one_hot_encoder(df, df_name="application", training=training, nan_as_category=nan_as_category)

    # to_print = pd.concat([df1, df]).drop_duplicates(keep=False)
    # print("HERE", to_print.empty)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # del test_df
    gc.collect()
    return df


def bureau_and_balance(input_path, num_rows=None, nan_as_category=True, training=True):
    """
    Preprocess bureau.csv and bureau_balance.csv

    :param input_path: the path to the file " bureau.csv" and "bureau_balance.csv"
    :param num_rows: (int or None) if we do not want to read all the csv file but only a certain number of rows
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :param training:
    :return:
    :rtype: DataFrame
    """
    bureau = pd.read_csv(input_path + 'bureau.csv', nrows=num_rows)
    bb = pd.read_csv(input_path + 'bureau_balance.csv', nrows=num_rows)
    # bb1, bb_cat1 = one_hot_encoder(bb, nan_as_category)
    # bureau1, bureau_cat1 = one_hot_encoder(bureau, nan_as_category)

    bb, bb_cat = one_hot_encoder(bb, df_name="bureau_balance", training=training, nan_as_category=nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, df_name="bureau", training=training, nan_as_category=nan_as_category)

    # to_print1 = pd.concat([bb1, bb]).drop_duplicates(keep=False)
    # print("HERE", to_print1.empty)

    # to_print2 = pd.concat([bureau1, bureau]).drop_duplicates(keep=False)
    # print("HERE", to_print2.empty)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg


def previous_applications(input_path, num_rows=None, nan_as_category=True, training=True):
    """
    Preprocess previous_applications.csv

    :param input_path: (string) the path to the file "previous_application.csv"
    :param num_rows: (int or None) if we do not want to read all the csv file but only a certain number of rows
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :param training:
    :return:
    :rtype: DataFrame
    """
    prev = pd.read_csv(input_path + 'previous_application.csv', nrows=num_rows)
    # prev1, cat_cols1 = one_hot_encoder(prev, nan_as_category=nan_as_category)
    prev, cat_cols = one_hot_encoder(prev, df_name="previous_application", training=training,
                                     nan_as_category=nan_as_category)

    # to_print = pd.concat([prev1, prev]).drop_duplicates(keep=False)
    # print("HERE", to_print.empty)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


def pos_cash(input_path, num_rows=None, nan_as_category=True, training=True):
    """
    Preprocess POS_CASH_balance.csv

    :param input_path: (string) the path to the file "POS_CASH_balance.csv"
    :param num_rows: (int or None) if we do not want to read all the csv file but only a certain number of rows
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :param training:
    :return:
    :rtype: DataFrame
    """
    pos = pd.read_csv(input_path + 'POS_CASH_balance.csv', nrows=num_rows)
    # pos1, cat_cols1 = one_hot_encoder(pos, nan_as_category=nan_as_category)
    pos, cat_cols = one_hot_encoder(pos, df_name="POS_CASH_balance", training=training,
                                    nan_as_category=nan_as_category)

    # to_print = pd.concat([pos1, pos]).drop_duplicates(keep=False)
    # print("HERE", to_print.empty)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg


def installments_payments(input_path, num_rows=None):
    """
    Preprocess installments_payments.csv
    :param input_path: (string)
    :param num_rows: (int or None) if we do not want to read all the csv file but only a certain number of rows
    :return:
    :rtype: (DataFrame)
    """
    ins = pd.read_csv(input_path + 'installments_payments.csv', nrows=num_rows)
    # ins1, cat_cols1 = one_hot_encoder(ins, nan_as_category=nan_as_category)
    # ins, cat_cols = one_hot_encoder_2(ins, df_name="installments_payments", training=training,
    #                                 nan_as_category=nan_as_category)
    # print("HERE CAT COL,", cat_cols)

    # to_print = pd.concat([ins1, ins]).drop_duplicates(keep=False)
    # print("HERE", to_print.empty)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    # for cat in cat_cols: # TODO delete ??
    #   aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


def credit_card_balance(input_path, num_rows=None, nan_as_category=True, training=True):
    """
    Preprocess credit_card_balance.csv

    :param input_path: (string) the path to the file "credit_card_balance.csv"
    :param num_rows: (int or None) if we do not want to read all the csv file but only a certain number of rows
    :param nan_as_category: (True or False) to get dummies (one hot encoder) with a NaN column
    :param training:
    :return:
    :rtype: DataFrame
    """
    cc = pd.read_csv(input_path + 'credit_card_balance.csv', nrows=num_rows)
    # cc1, cat_cols1 = one_hot_encoder(cc, nan_as_category=nan_as_category)
    cc, cat_cols = one_hot_encoder(cc, df_name="credit_card_balance", training=training,
                                   nan_as_category=nan_as_category)

    # to_print = pd.concat([cc1, cc]).drop_duplicates(keep=False)
    # print("HERE", to_print.empty)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def generate_dataset(input_path, application_filename, output_file, training=True, debug=False):
    """
    Main function : orchestrates the generation of the dataset such as named in the output_file parameter.

    :param input_path: (string) the root path to all the csv files to preprocess
    :param application_filename: (string) name of csv file
    :param output_file: (string) path + filename
    :param training:
    :param debug: (True or False) if debug is True, then we only rad 10000 rows
    :return: None
    :rtype: None
    """
    num_rows = 10000 if debug else None
    df = application(input_path, application_filename, num_rows, training=training)
    with timer("1) Process bureau and bureau_balance"):
        bureau = bureau_and_balance(input_path, num_rows, training=training)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("2) Process previous_applications"):
        prev = previous_applications(input_path, num_rows, training=training)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("3) Process POS-CASH balance"):
        pos = pos_cash(input_path, num_rows, training=training)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("4) Process installments payments"):
        ins = installments_payments(input_path, num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("5) Process credit card balance"):
        cc = credit_card_balance(input_path, num_rows, training=training)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
    with timer("6) Saving cleaned dataset"):
        print("Cleaned dataset shape :", df.shape)
        print("Setting client's id as index")  # TODO check that this works !!
        df = df.set_index("SK_ID_CURR")
        print("Re-indexed dataset shape :", df.shape)
        df.to_csv(output_file, index_label="SK_ID_CURR", sep=",")  # read with index_col = "SK_ID_CURR"
        gc.collect()  # delete ??


# TODO save csv with less disk space + faster read_csv
# https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv
# https://stackoverflow.com/questions/55299536/pandas-dataframe-csv-reduce-disk-size

# select features ? / remove features that are not important for the decision ?

if __name__ == "__main__":
    with timer("Full model run"):
        generate_dataset(input_path="dataset/source/", application_filename='application_train.csv',
                         output_file="dataset/cleaned/data_train_preprocessed_vf.csv", training=True)
        generate_dataset(input_path="dataset/source/", application_filename='application_test.csv',
                         output_file="dataset/cleaned/data_test_preprocessed_vf.csv", training=False)
        # preprocessed_one_query = generate_dataset(input_path="dataset/cleaned/",
        #                                         application_filename='one_query_test.csv',
        #                                        output_file="dataset/cleaned/preprocessed_one_query_test.csv",
        #                                       training=False)
