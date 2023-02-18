import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import RdYlGn

from utils import *

import numpy as np
import shap
import matplotlib
import seaborn as sns


############# GAUGE#####################"""

def rectangle_gauge(client_id, client_probability, threshold):
    """Draws a gauge for the result of credit application, and an arrow at the client probability of default.
    Args :
    - id (int) : client ID.
    - client_probability (float).
    Returns :
    - draws a matplotlib figure.
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 1))
    fig.suptitle(f"Client {client_id}: probability of credit default (%)",
                 size=15,
                 y=1.1)
    ax.add_patch(
        Rectangle((0, 0),
                  width=threshold * 100,
                  height=1,
                  color=(0.5, 0.9, 0.5, 0.5)))
    ax.add_patch(
        Rectangle((threshold * 100, 0),
                  width=100 - threshold * 100,
                  height=1,
                  color=(1, 0, 0, 0.5)))
    ax.plot((threshold * 100, threshold * 100), (0, 1),
            color='#FF8C00',
            ls=(0, (0.5, 0.5)),
            lw=6)
    ax.add_patch(
        FancyArrowPatch((client_probability * 100, 0.75),
                        (client_probability * 100, 0),
                        mutation_scale=20))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(0, 105, 10))
    ax.set_yticks([])
    st.pyplot(fig)


######################################## SHAP

def feature_description(feature, df_description):
    """Returns a description of the feature, taken from the table HomeCredit_columns_description.csv.
    Args :
    - feature (string).
    Returns :
    - its description (string.)
    """
    if feature in list(df_description.Row):
        description = df_description[df_description.Row ==
                                     feature]['Description'].iloc[0]
    else:
        description = "Description not available"
    return description


def shap_barplot(df_shap, df_description):
    """Plots an horizontal barplot of 10 SHAP values (the 5 most positive contributions and the 5 most negatives to the probability of default)
    Args :
    - df_shap (dataframe) : SHAP values and feature names.
    Returns :
    - matplotlib plot via st.pyplot.
    """
    # Preparation of data
    df = df_shap.sort_values(by='SHAP value', ascending=False)

    df_head = df.head(5).copy()
    df_tail = df.tail(5).copy()
    df = pd.concat([df_head, df_tail])
    # df = pd.concat([df.head(5), df.tail(5)])
    # df = df.head(5).append(df.tail(5)).copy()

    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4, figsize=(12, 5))
    colors = [RdYlGn(0.05 * i) for i in range(5)] + \
             [RdYlGn(0.8 + 0.04 * i) for i in range(5)]
    plt.barh(width=df['SHAP value'], y=df['feature'], color=colors)
    plt.xlabel('SHAP value')
    plt.ylabel('Features (top 5 contributors, both ways)')
    fig.suptitle('Impact on model output (credit default)', y=0.92, size=14)
    st.pyplot(fig)

    st.caption(
        "Horizontal scale : contribution to log odds of credit default.")
    # adds a list with the description of each feature on the graph
    add_feature_description(list_features=list(df['feature']), df_description=df_description)


def add_feature_description(list_features, df_description):
    """
        # adds a list with the description of each feature on the graph
    """
    with st.expander("Features description", expanded=False):
        for feature in list_features:
            st.caption(feature + ": " + feature_description(feature, df_description))

            ###################################################################################


def shap_force_plot(shap_values, expected_value, prediction, client_df):
    """
    :expected_value: (array)
    """
    plt.rcParams['figure.autolayout'] = True

    fig = shap.force_plot(np.around(expected_value[prediction], decimals=2),
                          np.around(shap_values[prediction], decimals=2),
                          np.around(client_df, decimals=2),
                          matplotlib=True,
                          show=False,
                          text_rotation=15,
                          figsize=(8, 4))
    fig.suptitle(
        "Shows which features had the most influence on the model's prediction for a single observation. \n Features in red increase the probability while blue ones decrease it",
        y=0.92, size=20)

    # to see the plot in notebook :
    # plt.show()
    # to see the plot in streamlit dashboard
    st.pyplot(fig)


def shap_summary_plot(shap_values, client_df):
    # Plotting
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(figsize=(12, 5), edgecolor='black', linewidth=4)  # summary_plot returns None

    # we need to prepare a fig for matplotlib to display the graph

    shap.summary_plot(shap_values, features=client_df, feature_names=client_df.columns)
    # fig.suptitle(
    #   "Shows the average impact of the features on the model output",
    #  y=0.92, size=20)
    # to see the plot in notebook :
    # plt.show()
    # to see the plot in streamlit dashboard
    st.pyplot(fig)

    ###############################################################################################


def global_feature_importance_barplot(dict_f_i, df_description, max_features_to_display=20):
    """
    TODO : model.feature_importances_ works only for LGBM ??? to check and clean code
    """
    # return barplot
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(figsize=(20, 12))  # for seaborn !!
    matplotlib.rc('ytick', labelsize=15)
    matplotlib.rc('xtick', labelsize=15)

    # for seaborn, "fig = " doesn't work !
    # to not display all of the 777 features, we display only the 15 or 20 most important ones
    list_features = list(dict_f_i.keys())[:max_features_to_display]
    values_per_feature = list(dict_f_i.values())[:max_features_to_display]

    bar_plot = sns.barplot(x=values_per_feature,
                           y=list_features)  # orient='h'

    bar_plot.axes.set_title("Global Feature importance of the model used for the prediction"
                            , fontsize=50)
    bar_plot.set_xlabel("Value of global feature importance", fontsize=30)
    bar_plot.set_ylabel("Feature", fontsize=20)

    # to see the plot in notebook :
    # plt.show()
    # to see the plot in streamlit dashboard
    st.pyplot(fig)

    add_feature_description(list_features=list_features, df_description=df_description)


###########################################################################""

#### BIVARIATE GRAPH

def contourplot(feature_1, feature_2, client_df, client_id, df_all_clients, df_description):
    """Contour plot for the observed probability of default as a function of 2 features.
    Args :
    - feature1 (string).
    - feature2 (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    figure = contourplot_in_common(df_all_clients, feature_1, feature_2)
    x_client = client_df[feature_1].iloc[0]
    y_client = client_df[feature_2].iloc[0]
    if str(x_client) == "nan" or str(y_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        y_center = (plt.ylim()[1] + plt.ylim()[0]) / 2
        plt.text(s=f" Client {client_id}\n  data not available",
                 x=x_center,
                 y=y_center)
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=1)
        plt.axhline(y=y_client,
                    xmin=-1e10,
                    xmax=1e10,
                    c='k',
                    ls='dashed',
                    lw=1)
        # if I want to interpolate data : https://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines
    # plt.show()
    st.pyplot(figure)
    st.caption(feature_1 + ": " + feature_description(feature_1, df_description))
    st.caption(feature_2 + ": " + feature_description(feature_2, df_description))


def contourplot_in_common(df_all_clients, feature_1, feature_2):
    """Contour plot for the observed probability of default as a function of 2 features. Common to all clients.
    Args :
    - feature1 (string).
    - feature2 (string).
    Returns :
    - matplotlib figure.
    """
    target_mesh_size = 500  # target population for each mesh

    # Preparation of data
    df = pd.DataFrame({
        feature_1: df_all_clients[feature_1],
        feature_2: df_all_clients[feature_2],
        'y_true': df_all_clients["TARGET"]
    })
    df = df.dropna().copy()
    n_values = len(df)
    n_bins = int(np.ceil(np.sqrt(n_values / target_mesh_size)))
    bin_size = int(np.floor(n_values / n_bins))
    index_bin_start = sorted([bin_size * n for n in range(n_bins)])
    ser1 = df[feature_1].sort_values().copy()
    ser2 = df[feature_2].sort_values().copy()

    # Filling the grid
    grid_proba_default = np.full((n_bins, n_bins), -1.0)
    ser_true0 = (df['y_true'] == 0)
    ser_true1 = (df['y_true'] == 1)
    for i1, ind1 in enumerate(index_bin_start):
        for i2, ind2 in enumerate(index_bin_start):
            ser_inside_this_mesh = (df[feature_1] >= ser1.iloc[ind1]) & (df[feature_2] >= ser2.iloc[ind2]) & (
                    df[feature_1] <= ser1.iloc[ind1 + bin_size - 1]) & (
                                           df[feature_2] <= ser2.iloc[ind2 + bin_size - 1])
            # sum of clients true0 inside this square bin
            sum_0 = (ser_inside_this_mesh & ser_true0).sum()
            sum_1 = (ser_inside_this_mesh & ser_true1).sum()
            sum_ = sum_0 + sum_1
            if sum_ == 0:
                proba_default = 1
            else:
                proba_default = sum_1 / sum_
            grid_proba_default[i2, i1] = proba_default

    # X, Y of the grid
    X = [ser1.iloc[i + int(bin_size / 2)] for i in index_bin_start]
    Y = [ser2.iloc[i + int(bin_size / 2)] for i in index_bin_start]

    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.contourf(X, Y, grid_proba_default, cmap='Reds')
    plt.colorbar(shrink=0.8)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    fig.suptitle(
        f'Observed probability of default as a function of {feature_1} and {feature_2}',
        y=0.92)
    return fig


###########################################################################""

def boxplot_all_clients_compared_to_client_feature_value(data_all_clients, list_features, client_df):
    """
    Positions the client
    """
    mapping_x_ticks = {
        '1': 'Default Client',
        '0': 'Non Default Client'
    }
    fig = plt.figure(figsize=(12, 5))
    n = len(list_features)

    for i, feature in enumerate(list_features):
        # to display the boxplots on the same row
        position = int('1{}{}'.format(n, i + 1))
        ax = fig.add_subplot(position)

        feature_value = client_df[feature].values[0]  # we get the value for the client's feature

        # fig, ax = plt.subplots(figsize=(12, 9))
        # create boxplot
        bp = sns.boxplot(data=data_all_clients,
                         y=feature,
                         x='TARGET',
                         orient="v",
                         showfliers=False,
                         palette=["#4286DE", "#EA365B"],
                         ax=ax)
        # add client threshold
        bp.axhline(feature_value,
                   color='r',
                   label='Client value'
                   )
        # add label and legend
        bp.legend()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels = [mapping_x_ticks[i] for i in labels]
        bp.set_xticklabels(labels)
        bp.set_title(f'{feature}')
        bp.title.set_size(20)

        # to see the plot in notebook :
        # plt.tight_layout()
        # plt.show()
        # to see the plot in streamlit dashboard
    plt.tight_layout()
    st.pyplot(fig)


###########################################""


def histgram_compared_to_all_clients(df_all_clients, feature, client_df):
    # we get the value for the client's feature
    feature_value = client_df[feature].values[0]

    fig = plt.figure(figsize=(12, 5))

    # Accepted clients TARGET == 0
    ax = fig.add_subplot(121)
    bp = sns.histplot(data=df_all_clients[df_all_clients["TARGET"] == 0],
                      x=feature,
                      bins=20)
    # add client threshold
    bp.axvline(feature_value,
               color='r',
               label='Client value'
               )
    plt.title("Distribution of {} for accepted clients".format(feature))

    # Refused clients TARGET == 1
    ax = fig.add_subplot(122)
    bp = sns.histplot(data=df_all_clients[df_all_clients["TARGET"] == 1],
                      x=feature, bins=20)
    # add client threshold
    bp.axvline(feature_value,
               color='r',
               label='Client value'
               )
    plt.title('Distribution of {} for refused clients'.format(feature))

    plt.tight_layout()
    st.pyplot(fig)


###########################################################################""

def add_new_client_to_data_all_clients(data_all_clients, df_client, prediction, y_label="TARGET"):
    """
    # TODO : rename function var to not have "metier logic" ?
    :param:
    :param:
    :param:
    :param:
    :return:
    :rtype:
    """
    # 1) we add the prediction to the client's dataframe (we add one column
    # df_client[y_label] = pred # this adds the column at the end
    print("__Dataframe shape before adding new client :", data_all_clients.shape)

    # we need the column with the prediction at the beginning
    df_client.insert(0, y_label, prediction, True)  # first column / column_name / value / inplace

    # 2) we add the new client to the dataset of all clients
    df_merged = pd.concat([df_client, data_all_clients])
    print("__Dataframe shape after adding new client :", df_merged.shape)
    return df_merged

########################################################################################################################
