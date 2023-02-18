from utils import *
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import RdYlGn
import numpy as np
import shap
import matplotlib
import seaborn as sns


def rectangle_gauge(client_id, client_probability, threshold):
    """
    Draws a gauge for the result of credit application, and an arrow at the client probability of default.

    :param client_id: (int)
    :param client_probability: (float)
    :param threshold:
    :return: None
    :rtype: None
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


def feature_description(feature, df_description):
    """
    Returns a description of the feature, taken from the table HomeCredit_columns_description.csv.

    :param feature: (string)
    :param df_description: (string)
    :return:
    :rtype:
    """
    if feature in list(df_description.Row):
        description = df_description[df_description.Row ==
                                     feature]['Description'].iloc[0]
    else:
        description = "Description not available"
    return description


def shap_barplot(df_shap, df_description):
    """
    Plots an horizontal barplot of 10 SHAP values (the 5 most positive contributions and the 5 most negatives to the probability of default)

    :param df_shap: (dataframe) : SHAP values and feature names
    :param df_description:
    :return:
    :retype: None
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
    Adds a list with the description of each feature on the graph

    :param list_features:
    :param df_description:
    :return: None
    :rtype: None
    """
    with st.expander("Features description", expanded=False):
        for feature in list_features:
            st.caption(feature + ": " + feature_description(feature, df_description))


def shap_force_plot(shap_values, expected_value, prediction, client_df):
    """

    :param shap_values:
    :param expected_value: (array)
    :param prediction:
    :param client_df:
    :return:
    :rtype:
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
        y=0.92, size=15)

    # to see the plot in notebook :
    # plt.show()
    # to see the plot in streamlit dashboard
    st.pyplot(fig)


def shap_summary_plot(shap_values, client_df):
    """

    :param shap_values:
    :param client_df:
    :return:
    :rtype:
    """
    # Plotting
    plt.rcParams['figure.autolayout'] = True
    fig = plt.figure(figsize=(12, 5), edgecolor='black', linewidth=4)  # summary_plot returns None

    # we need to prepare a fig for matplotlib to display the graph
    shap.summary_plot(shap_values, features=client_df, feature_names=client_df.columns)

    # to see the plot in notebook :
    # plt.show()
    # to see the plot in streamlit dashboard
    st.pyplot(fig)


def global_feature_importance_barplot(dict_f_i, df_description, max_features_to_display=20):
    """
    TODO : model.feature_importances_ works only for LGBM ??? to check and clean code
    :param dict_f_i:
    :param df_description:
    :param max_features_to_display:
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

def boxplot_all_clients_compared_to_client_feature_value(data_all_clients, feature, client_df):
    """
    Positions the client

    :param data_all_clients:
    :param feature:
    :param client_df:
    """
    mapping_x_ticks = {
        '1': 'Client refused for a loan',
        '0': 'Client accepted for a loan'
    }
    # fig = plt.figure(figsize=(12, 5))
    fig, ax = plt.subplots(figsize=(10, 5))
    # n = len(list_features)

    # for i, feature in enumerate(list_features):
    # to display the boxplots on the same row
    #   position = int('1{}{}'.format(n, i + 1))
    #  ax = fig.add_subplot(position)

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
    # bp.legend()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [mapping_x_ticks[i] for i in labels]
    bp.set_xticklabels(labels)

    bp.set_title(f'Box plots for {feature}')
    bp.title.set_size(20)

    # to see the plot in notebook :
    # plt.tight_layout()
    # plt.show()
    # to see the plot in streamlit dashboard
    plt.tight_layout()
    st.pyplot(fig)


def histgram_compared_to_all_clients(df_all_clients, feature, client_df):
    """

    :param df_all_clients:
    :param feature:
    :param client_df:
    """
    # we get the value for the client's feature
    feature_value = client_df[feature].values[0]

    fig = plt.figure(figsize=(12, 5))

    # Accepted clients TARGET == 0
    ax = fig.add_subplot(121)
    bp = sns.histplot(data=df_all_clients[df_all_clients["TARGET"] == 0],
                      x=feature,
                      bins=20,
                      color="#4286DE")
    # add client threshold
    bp.axvline(feature_value,
               color='r',
               label='Client value'
               )
    plt.title("Distribution of {} for accepted clients".format(feature))

    # Refused clients TARGET == 1
    ax = fig.add_subplot(122)
    bp = sns.histplot(data=df_all_clients[df_all_clients["TARGET"] == 1],
                      x=feature, bins=20, color="#EA365B")
    # add client threshold
    bp.axvline(feature_value,
               color='r',
               label='Client value'
               )
    plt.title('Distribution of {} for refused clients'.format(feature))

    plt.tight_layout()
    st.pyplot(fig)


def add_position_client(client_df, feature_1, feature_2):
    x_client = client_df[feature_1].iloc[0]
    y_client = client_df[feature_2].iloc[0]

    if str(x_client) == "nan" or str(y_client) == "nan":
        x_center = (plt.xlim()[1] + plt.xlim()[0]) / 2
        y_center = (plt.ylim()[1] + plt.ylim()[0]) / 2
        plt.text(s=f"Data not available for the selected client",
                 x=x_center,
                 y=y_center)  # ,
        # ax=ax)

    # plot only the y-axis line
    if str(y_client) != "nan":
        plt.axhline(y=y_client,
                    c='k',
                    ls='dashed',
                    lw=1)  # ,
        # ax=ax)

    # plot only the x-axis line
    if str(x_client) != "nan":
        plt.axvline(x=x_client,
                    c='k',
                    ls='dashed',
                    lw=1)  # ,
        # ax=ax)


def scatterplot_comparing_all_clients(df_all_clients, feature_1, feature_2, client_df):
    """

    :param df_all_clients:
    :param feature_1:
    :param feature_2:
    :param client_df:
    """
    if feature_1 != feature_2:
        fig = plt.figure(figsize=(8, 6))

        plot = sns.scatterplot(x=feature_1, y=feature_2, data=df_all_clients, style='TARGET', hue='TARGET',
                               size_order=[1, 0], size="TARGET", palette=["#4286DE", "#EA365B"])
        add_position_client(client_df, feature_1, feature_2)
        plot.set_title("Scatter plot of {} as a function of {} for all clients".format(feature_2, feature_1))

        plt.tight_layout()
        # plt.show()
        st.pyplot(fig)
    else:
        st.write("Choose two different features")
