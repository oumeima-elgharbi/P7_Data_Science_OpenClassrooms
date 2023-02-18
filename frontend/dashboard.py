# Importing librairies
from graphs_dashboard import *
from dashboard_requests import *
from utils import *

import gc
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

#############################################################################
# TODO : clean code / download folder when building the app

import os
from os import listdir

this_dir = os.getcwd()
all_files = [f for f in listdir(this_dir)]

# print("THIS CWD", this_dir)
# print("ALL FILES : ", all_files)

if "resources" not in all_files:
    import subprocess

    print("__Download resources folder__")
    subprocess.call(r'python script_download_data_folder.py', shell=True)
    gc.collect()

##############################################################################

# Config :Check host if deployment or not

print("_____Getting config_____")
config = read_yml("config.yml")

print("Deployment ? {}".format(config["deploy"]["is"]))
if config["deploy"]["is"]:
    HOST = config["deploy"]["prod_api"]
else:
    HOST = config["deploy"]["dev"]

# Config front-end
print("_____Getting config front-end_____")
config_front = read_yml("config_frontend.yml")

THRESHOLD = config_front["threshold"]
ENDPOINT_PREDICT = config_front["endpoints"]["endpoint_predict"]
ENDPOINT_SHAP = config_front["endpoints"]["endpoint_shap"]
ENDPOINT_CLIENT_DATA = config_front["endpoints"]["endpoint_client_data"]
ENDPOINT_SHAP_EXPECTED = config_front["endpoints"]["endpoint_shap_expected"]  # TODO clean
ENDPOINT_FEATURE_IMPORTANCE = config_front["endpoints"]["endpoint_feature_importance"]
DATA_ALL_CLIENTS_PATH = config_front["known_clients_database_preprocessed"]
DF_DESCRIPTION = pd.read_csv(config_front["columns_description"], encoding="ISO-8859-1")  # not encoded in utf-8

# Initializing all the global variables
global CLIENT_ID
CLIENT_ID = 100005  # 456250

global NB_FEATURES_TO_PLOT
NB_FEATURES_TO_PLOT = 2

global CLIENT_JSON
global CLIENT_DF
global LIST_FEATURES
global SELECTED_FEATURE
global FEATURE_1
global FEATURE_2
global PREDICTION

# we initialize the first webview to homepage
global DASHBOARD_CHOICE
DASHBOARD_CHOICE = "Homepage"


def initialize_webview():
    """
     Initialize webview with sidebar to choose client and which dashboard to display

    :param: None
    :return: None
    :rtype: None
    """
    global DASHBOARD_CHOICE
    global CLIENT_ID

    # Load data
    # Default settings. This must be the first Streamlit command used in your app, and must only be set once.
    st.set_page_config(page_title="Project 7 Dashboard",
                       initial_sidebar_state="expanded",
                       layout="wide")

    # Sidebar
    with st.sidebar:
        logo_homecredit = Image.open('./img/Home-Credit-Logo.jpg')
        st.image(logo_homecredit, width=300)

        # Dashboard selector
        # st.write('## Menu :')
        DASHBOARD_CHOICE = st.radio('Menu', [
            'Homepage', 'Client dashboard', 'Comparing with previous clients'])  # label_visibility
        st.write('## ')
        st.write('## ')

        if DASHBOARD_CHOICE in ['Client dashboard', 'Comparing with previous clients']:
            # Client selector
            # st.write('## Client ID:')

            CLIENT_ID = st.number_input("Enter client ID", value=100005)
            st.caption("Example of client predicted negative (no default) : 100005")
            st.caption("Example of client predicted positive (credit default) : 456250")
            st.caption(" ")


###################################################################################

def homepage():
    """
     Homepage with an explanation about the context

    :param: None
    :return: None
    :rtype: None
    """
    st.title("Home Credit Default Risk Prediction")
    " "
    " "
    "This webview contains an **interactive dashboard** to explain to the bank's customers the reason of **approval or refusal of their credit application.**"
    "The probability of credit default has been calculated by a prediction model based on machine learning."
    " "
    " "
    "There are two views available :"
    "- A **Client dashboard** version that contains information about the acceptance or the refusal of the client's loan."
    "=> Global Feature Importance"
    "=> Local Feature Importance"
    " "
    "- And **Comparing with previous clients** which is a dashboard that compares the client to previous clients for whom we know if they have repaid or not their loan."
    " "
    "=> Distribution for a chosen feature"
    "=> Positioning the client compared to previous clients based on two selected features"


def initialize_client_dashboard():
    """

    :param: None
    :return: None
    :rtype: None
    """
    global CLIENT_ID
    global CLIENT_JSON
    global CLIENT_DF

    # Main title of the dashboard
    st.title(f'Default Risk Prediction for client {CLIENT_ID}')

    # Get client data
    try:
        CLIENT_JSON = request_client_data(HOST + ENDPOINT_CLIENT_DATA, CLIENT_ID)
        CLIENT_DF = json_to_df(CLIENT_JSON)
    except Exception as e:
        print("Exception raised while trying to get client data :\n\n", e)
        st.write('The client with the id {} is not in the database.'.format(CLIENT_ID))
        return  # to get out of the function


###############################################################

def basic_dashboard():
    """
    Basic dashboard with prediction and local feature importance

    :param: None
    :return: None
    :rtype: None
    """
    initialize_client_dashboard()

    st.header('Result of credit application')
    proba_view()

    st.header("Global Feature Importance")
    global_feature_importance_view()
    shap_summary_plot_view()
    "---------------------------"
    st.header("Local Feature Importance")
    shap_barplot_view()
    shap_force_plot_view()


def proba_view():
    """
     Result of credit application

    :param: None
    :return: None
    :rtype: None
    """
    global PREDICTION
    try:
        probability = request_prediction(HOST + ENDPOINT_PREDICT, CLIENT_JSON)
        if probability < THRESHOLD:
            PREDICTION = 0
            st.success(
                f"  \n __CREDIT ACCEPTED__  \n  \nThe probability of default of the applied credit is __{round(100 * probability, 1)}__% (lower than the threshold of {100 * THRESHOLD}% for obtaining the credit).  \n "
            )
        else:
            PREDICTION = 1
            st.error(
                f"__CREDIT REFUSED__  \nThe probability of default of the applied credit is __{round(100 * probability, 1)}__% (higher than the threshold of {100 * THRESHOLD}% for obtaining the credit).  \n "
            )
        if st.button('Predict'):  # predict_btn = st.button('Predict') # if predict_btn:
            rectangle_gauge(CLIENT_ID, probability, THRESHOLD)
    except Exception as e:
        print("Exception raised :", e)
        st.write("Couldn't compute probability of loan for client {}...".format(CLIENT_ID))


def shap_barplot_view():
    """
    Local SHAP barplot

    :param: None
    :return: None
    :rtype: None
    """
    # st.write('')
    try:
        # get shap values for the selected client
        client_shap_json = request_shap(HOST + ENDPOINT_SHAP, CLIENT_JSON)
        df_shap = json_to_df(client_shap_json)  # just need pd.Dataframe()
        shap_barplot(df_shap, DF_DESCRIPTION)
    except Exception as e:
        print("Exception raised :", e)


def shap_force_plot_view():
    """
    Local SHAP force plot

    :param: None
    :return: None
    :rtype: None
    """
    st.write('SHAP Force plot')
    try:
        # get shap values for the selected client
        response_shap = request_shap_expected(HOST + ENDPOINT_SHAP_EXPECTED, CLIENT_JSON)

        expected_value = response_shap["expected_value"]  # CF back end doc TODO refacto
        encoded_numpy_data = response_shap["shap_values"]

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decoded_arrays = json.loads(encoded_numpy_data)
        final_numpy_array = np.asarray(decoded_arrays["array"])

        shap_force_plot(final_numpy_array, expected_value, PREDICTION, CLIENT_DF)
    except Exception as e:
        print("Exception raised :", e)


def global_feature_importance_view():
    """
    :param: None
    :return: None
    :rtype: None
    """
    global LIST_FEATURES
    st.write('Global Feature Importance of the model used for the prediction')
    try:
        # get global feature importance
        response = request_feature_importance(HOST + ENDPOINT_FEATURE_IMPORTANCE, CLIENT_JSON)
        global_feature_importance_barplot(response, DF_DESCRIPTION)

        # here we keep the value ?? for the boxplot ??
        LIST_FEATURES = list(response.keys())  # the response is a dict {feature: value}

    except Exception as e:
        print("Exception raised :", e)


def shap_summary_plot_view():  # TODO refacto ??
    """

    :param: None
    :return: None
    :rtype: None
    """
    st.write('SHAP Feature Importance summary plot')
    try:
        # get shap values for the selected client
        response_shap = request_shap_expected(HOST + ENDPOINT_SHAP_EXPECTED, CLIENT_JSON)
        encoded_numpy_data = response_shap["shap_values"]

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decoded_arrays = json.loads(encoded_numpy_data)
        final_numpy_array = np.asarray(decoded_arrays["array"])
        shap_values = list(final_numpy_array)  # TODO refacto / response is an array (2, 1, 777)
        # or shap_values should be a list of 2 arrays, not an array of two arrays

        st.write("Shows the average impact of the features on the model output")
        shap_summary_plot(shap_values, CLIENT_DF)

    except Exception as e:
        print("Exception raised :", e)


#################################################################################""


def advanced_dashboard():
    """
    The view contains information about the client compared to other clients in the database
    Position of the client vs other clients

    :param: None
    :return: None
    :rtype: None
    """
    initialize_client_dashboard()
    global NB_FEATURES_TO_PLOT
    global SELECTED_FEATURE
    global FEATURE_1
    global FEATURE_2

    # to get a list of important features
    global LIST_FEATURES
    try:
        # get global feature importance
        response = request_feature_importance(HOST + ENDPOINT_FEATURE_IMPORTANCE, CLIENT_JSON)
        # here we keep the value ?? for the boxplot ??
        LIST_FEATURES = list(response.keys())  # the response is a dict {feature: value}
    except Exception as e:
        print("Exception raised :", e)
        return

    sidebar_nb_features()

    # Distribution plots
    selectbox_for_distribution_plots()
    boxplot_view()
    hist_view()

    # Bi variate plot
    selectbox_for_scatterplot()
    scatter_plot_view()


def sidebar_nb_features():
    """

    :param: None
    :return: None
    :rtype: None
    """
    global NB_FEATURES_TO_PLOT
    global LIST_FEATURES

    st.sidebar.write(' ')
    NB_FEATURES_TO_PLOT = int(st.sidebar.selectbox(
        'Number of features to choose from',
        [2, 3, 4, 5, 6, 7, 8, 9, 10]))

    st.sidebar.write(' ')

    # this is to have a pretty display of all the features in the sidebar
    s = ""
    for e in LIST_FEATURES[:NB_FEATURES_TO_PLOT]:
        s = s + '\n- ' + e

    st.sidebar.write(
        f'*{str(NB_FEATURES_TO_PLOT)} most impactful features for selected client :* {s}')
    st.sidebar.write(' ')  # to add a blank space


def selectbox_for_distribution_plots():
    """

    :param: None
    :return: None
    :rtype: None
    """
    global SELECTED_FEATURE
    global LIST_FEATURES
    global NB_FEATURES_TO_PLOT
    # Positioning of the client with comparison to other clients (choice of feature)
    "---------------------------"
    st.header(
        'Distribution of a selected feature'
    )
    SELECTED_FEATURE = st.selectbox(
        f'Choose a feature among {len(LIST_FEATURES[:NB_FEATURES_TO_PLOT])}',
        options=LIST_FEATURES[:NB_FEATURES_TO_PLOT])  # to have a maximum number of features to plot


def selectbox_for_scatterplot():
    """

    :param: None
    :return: None
    :rtype: None
    """
    global FEATURE_1
    global FEATURE_2
    global LIST_FEATURES
    global NB_FEATURES_TO_PLOT

    "---------------------------"
    st.header(
        'Comparing the client to previous clients using two selected features'
    )

    list_for_feature_1 = LIST_FEATURES[:NB_FEATURES_TO_PLOT]
    FEATURE_1 = st.selectbox(
        f'Choose the first feature among {len(list_for_feature_1)}',
        options=list_for_feature_1)  # to have a maximum number of features to plot

    # we remove the feature 1 from the choices for feature 2 ??
    list_for_feature_2 = list_for_feature_1.copy()
    list_for_feature_2.remove(FEATURE_1)
    FEATURE_2 = st.selectbox(
        f'Choose the second feature among {len(list_for_feature_2)}',
        options=list_for_feature_2)  # to have a maximum number of features to plot


def boxplot_view():
    """

    :param: None
    :return: None
    :rtype: None
    """
    st.write(
        'Box plots for the most importance features using all the known clients and comparing to the current client')

    # we read here the database using only the feature to plot and the index
    columns_list = [SELECTED_FEATURE, "SK_ID_CURR", "TARGET"]
    print("__Reading database of all clients for the list of features : ", columns_list)
    data_all_clients = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                                   usecols=columns_list)

    boxplot_all_clients_compared_to_client_feature_value(data_all_clients, SELECTED_FEATURE, CLIENT_DF)


def hist_view():
    """
    :param: None
    :return: None
    :rtype: None
    """
    st.write(
        'Histograms for the most importance features using all the known clients and comparing to the current client')

    # we read here the database using only the feature to plot and the index
    columns_list = [SELECTED_FEATURE, "SK_ID_CURR", "TARGET"]
    print("__Reading database of all clients for the list of features : ", columns_list)
    data_all_clients = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                                   usecols=columns_list)

    histgram_compared_to_all_clients(data_all_clients, SELECTED_FEATURE, CLIENT_DF)


def scatter_plot_view():
    """
    :param: None
    :return: None
    :rtype: None
    """
    # we read here the database using only the feature to plot and the index
    columns_list = [FEATURE_1, FEATURE_2, "SK_ID_CURR", "TARGET"]
    print("__Reading database of all clients for the list of features : ", columns_list)
    data_all_clients = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                                   usecols=columns_list)

    scatterplot_comparing_all_clients(data_all_clients, FEATURE_1, FEATURE_2, CLIENT_DF)


#############################################################################################"

def dashboard_view():
    """
    Displays the correct view as a function of the dashboard selected in the menu.

    :param: None
    :return: None
    :rtype: None
    """
    if DASHBOARD_CHOICE == 'Homepage':
        homepage()

    elif DASHBOARD_CHOICE == 'Client dashboard':
        basic_dashboard()
    else:
        advanced_dashboard()


######################################################################

def main():
    """
    :param: None
    :return: None
    :rtype: None
    """
    # Sidebar
    initialize_webview()

    # choose dashboard type
    dashboard_view()


if __name__ == '__main__':
    main()
