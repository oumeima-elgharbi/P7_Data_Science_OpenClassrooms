import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import RdYlGn

from utils import *


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
    fig = plt.figure(edgecolor='black', linewidth=4)
    colors = [RdYlGn(0.05 * i) for i in range(5)] + \
             [RdYlGn(0.8 + 0.04 * i) for i in range(5)]
    plt.barh(width=df['SHAP value'], y=df['feature'], color=colors)
    plt.xlabel('SHAP value')
    plt.ylabel('Features (top 5 contributors, both ways)')
    fig.suptitle('Impact on model output (credit default)', y=0.92, size=14)
    st.pyplot(fig)
    st.caption(
        "Horizontal scale : contribution to log odds of credit default.")
    with st.expander("Features description", expanded=False):
        for feature in list(df['feature']):
            st.caption(feature + ": " + feature_description(feature, df_description))


#######################################################################################################################
######################################################################"

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


import joblib
import numpy as np


def lineplot_in_common(data_all_clients, feature, y_label='TARGET'):
    """Line plot of a quantitative feature. Common to all clients.
    Plot smoothed over 4000 clients. One dot plotted every 1000 clients.
    Args :
    - feature (string).
    Returns :
    - matplotlib figure.
    """
    target_bin_size = 4000

    # preparation of data
    df = data_all_clients.copy()

    df = df.dropna().sort_values(axis=0, by=feature).copy()
    n_values = len(df)
    n_bins = int(np.ceil(n_values / target_bin_size))
    bin_size = int(np.floor(n_values / n_bins))
    index_bin_start = [bin_size * n for n in range(n_bins)] + [int(bin_size * (n + 0.25)) for n in range(n_bins)] \
                      + [int(bin_size * (n + 0.5)) for n in range(n_bins)] + \
                      [int(bin_size * (n + 0.75)) for n in range(n_bins)]
    index_bin_start = sorted(index_bin_start)

    # Observed probability of default for every bins
    proba_default = []
    feature_value_start = []
    for i in index_bin_start[2:-2]:
        some_bin = df.iloc[int(i - 0.5 * bin_size):int(i + 0.5 * bin_size)]
        some_bin_sum0 = (some_bin[y_label] == 0).sum()
        some_bin_sum1 = (some_bin[y_label] == 1).sum()
        some_bin_sum = some_bin_sum0 + some_bin_sum1
        proba_default_ = some_bin_sum1 / some_bin_sum
        proba_default.append(proba_default_)
        feature_value_start.append(df[feature].iloc[i])

    # Plotting
    plt.style.use('seaborn')
    fig = plt.figure(edgecolor='black', linewidth=4)
    plt.plot(feature_value_start, proba_default, color='k')
    ylim_high = plt.ylim()[1]
    plt.fill_between(x=feature_value_start, y1=proba_default, y2=0, color='r')
    plt.fill_between(x=feature_value_start,
                     y1=proba_default,
                     y2=1,
                     color='limegreen')
    plt.ylabel('Observed probability of default')
    plt.xlabel(feature)
    fig.suptitle(f'Observed probability of default as a function of {feature}',
                 y=0.92)
    plt.ylim(0, max(ylim_high, 0.3))
    return fig


def lineplot(data_all_clients, client_df, client_id, threshold, feature, df_description):
    """Plots a lineplot of the quantitative feature.
    Args :
    - feature (string).
    Returns :
    - matplotlib plot via st.pyplot.
    """
    # if feature in [
    #    'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'AMT_ANNUITY'
    # ]:
    #    figure = joblib.load('./resources/figure_lineplot_' + feature +
    #                         '_for_bankclerk.joblib')
    # else:
    # figure = lineplot_in_common(feature)
    figure = lineplot_in_common(data_all_clients, feature)
    y_max = plt.ylim()[1]
    x_client = client_df[feature].iloc[0]
    if str(x_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        plt.annotate(text=f" Client {client_id}\n  data not available",
                     xy=(x_center, 0),
                     xytext=(x_center, y_max * 0.9))
    else:
        plt.axvline(x=x_client,
                    ymin=-1e10,
                    ymax=1e10,
                    c='k',
                    ls='dashed',
                    lw=2)
        plt.axhline(y=threshold,
                    xmin=-1e10,
                    xmax=1e10,
                    c='darkorange',
                    ls='dashed',
                    lw=1)  # line for the optimum_threshold
        plt.annotate(text=f" Client {client_id}\n  {round(x_client, 3)}",
                     xy=(x_client, y_max * 0.9))
    st.pyplot(figure)
    st.caption(feature + ": " + feature_description(feature, df_description))
