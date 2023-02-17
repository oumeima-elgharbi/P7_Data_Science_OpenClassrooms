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
                          figsize=(20, 6))
    fig.suptitle(
        "Shows which features had the most influence on the model's prediction for a single observation. \n Features in red increase the probability while blue ones decrease it",
        y=0.92, size=20)

    # to see the plot in notebook :
    # plt.show()
    # to see the plot in streamlit dashboard
    st.pyplot(fig)


###############################################################################################

def global_feature_importance_barplot(columns, global_feature_importance, df_description, max_features_to_display=20):
    """
    TODO : model.feature_importances_ works only for LGBM ??? to check and clean code
    """
    # create dict {columns_name:model_feature_importance}
    dict_f_i = dict(zip(columns, global_feature_importance))  # model.feature_importances_
    # sorted by feature_importance
    dict_f_i = {k: v for k, v in sorted(dict_f_i.items(), key=lambda item: item[1], reverse=True)}
    # return barplot
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(figsize=(25, 14))  # for seaborn !!
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
    # to see the plot in streamlit dahsboard
    st.pyplot(fig)

    add_feature_description(list_features=list_features, df_description=df_description)


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
