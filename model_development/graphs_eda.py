import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(path, filename):
    """

    :param path:
    :param filename: (string)
    :return:
    :rtype: (DataFrame)
    """
    print("___Loading raw dataset___")

    # Load raw data
    dataset_file = "{}{}".format(path, filename)
    dataset = pd.read_csv(dataset_file, encoding="windows-1252")

    print("Initial shape :", dataset.shape)
    return dataset


def missing_data(data):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param data:
    :return:
    :rtype: (DataFrame)
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def plot_stats(df, feature, label_rotation=False, horizontal_layout=True):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param feature:
    :param label_rotation:
    :param horizontal_layout:
    :return: None
    :rtype: None
    """
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();


def plot_distribution(df, var):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param var:
    :return: None
    :rtype: None
    """

    i = 0
    t1 = df.loc[df['TARGET'] != 0]
    t0 = df.loc[df['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for feature in var:
        i += 1
        plt.subplot(2, 2, i)
        sns.kdeplot(t1[feature], bw=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();


def plot_b_stats(df, feature, label_rotation=False, horizontal_layout=True):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param feature:
    :param label_rotation:
    :param horizontal_layout:
    :return: None
    :rtype: None
    """
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();


def plot_b_distribution(df, feature, color):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param feature:
    :param color:
    :return: None
    :rtype: None
    """
    plt.figure(figsize=(10, 6))
    plt.title("Distribution of %s" % feature)
    sns.distplot(df[feature].dropna(), color=color, kde=True, bins=100)
    plt.show()


def is_outlier(points, thresh=3.5):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    # Source: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting (see references)
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def plot_b_o_distribution(df, feature, color):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param feature:
    :param color:
    :return:
    :rtype:
    """
    plt.figure(figsize=(10, 6))
    plt.title("Distribution of %s" % feature)
    x = df[feature].dropna()
    filtered = x[~is_outlier(x)]
    sns.distplot(filtered, color=color, kde=True, bins=100)
    plt.show()


# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_b_distribution_comp(df, var, nrow=2):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param var:
    :param nrow:
    :return: None
    :rtype: None
    """
    i = 0
    t1 = df.loc[df['TARGET'] != 0]
    t0 = df.loc[df['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow, 2, figsize=(12, 6 * nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow, 2, i)
        sns.kdeplot(t1[feature], bw=0.5, label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5, label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();


######################################################################################

def plot_p_stats(df, feature, label_rotation=False, horizontal_layout=True):
    """
    Source : https://www.kaggle.com/code/gpreda/home-credit-default-risk-extensive-eda#Explore-the-data

    :param df:
    :param feature:
    :param label_rotation:
    :param horizontal_layout:
    :return:
    :rtype:
    """
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    if (horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x=feature, y="Number of contracts", data=df1)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)

    s = sns.barplot(ax=ax2, x=feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if (label_rotation):
        s.set_xticklabels(s.get_xticklabels(), rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.show();
