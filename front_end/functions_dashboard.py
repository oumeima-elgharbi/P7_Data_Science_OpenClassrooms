# to find the parent directory

import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

# now we can import the module in the parent
# directory.

####################################

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import RdYlGn

from utils import *


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
