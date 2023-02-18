import requests
from functions_dashboard import *
from utils import *
import gc

import streamlit as st
import joblib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.cm import RdYlGn
from PIL import Image
from random import randint

from dashboard_requests import *

# SHAP : featu importance globale (constt) // local : le client 4 : telle var plus impacte sur son score et diff de feat importance
# global : 3e dans la lsite mais si client X : 1er revenu

# SMOTE : classes desequilibrées : dummy 0 : pour améliorer score sur classe 1 et réequilibrer dataset
# voir si ca améliore le score


#############################################################################
# TODO remove !! this is NOT clean code

import os
from os import listdir

this_dir = os.getcwd()
all_files = [f for f in listdir(this_dir)]

print("THIS CWD", this_dir)
print("ALL FILES : ", all_files)

if "resources" not in all_files:
    import subprocess

    print("__Download resources folder__")
    subprocess.call(r'python script_download_data_folder.py', shell=True)
    gc.collect()
##############################################################################

# 0) Config : unzip data and check host if deployment or not

print("_____Getting config_____")
config = read_yml("config.yml")

# print("__Unzip model and dataset__")
# unzip_file(path_to_zip_file=config["resources"]["zip"], directory_to_extract_to=config["resources"]["unzip"])

print("Deployment ? {}".format(config["deploy"]["is"]))
if config["deploy"]["is"]:
    HOST = config["deploy"]["prod_api"]
else:
    HOST = config["deploy"]["dev"]

##################################################################################################################

# 1) Config front-end
print("_____Getting config front-end_____")
config_front = read_yml("config_frontend.yml")

THRESHOLD = config_front["threshold"]
# ENDPOINT_GET_CLIENT_DATA = config_front["endpoints"]["endpoint_get_client_data"]
ENDPOINT_PREDICT = config_front["endpoints"]["endpoint_predict"]
ENDPOINT_SHAP = config_front["endpoints"]["endpoint_shap"]
ENDPOINT_CLIENT_DATA = config_front["endpoints"]["endpoint_client_data"]
DF_DESCRIPTION = pd.read_csv(config_front["columns_description"], encoding="ISO-8859-1")  # not encoded in utf-8

ENDPOINT_SHAP_EXPECTED = config_front["endpoints"]["endpoint_shap_expected"]  # TODO clean

ENDPOINT_FEATURE_IMPORTANCE = config_front["endpoints"]["endpoint_feature_importance"]

DATA_ALL_CLIENTS_PATH = config_front["known_clients_database_preprocessed"]

# TODO add dtype reading !!! https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv
#
# DATA_ALL_CLIENTS = pd.read_csv(config_front["known_clients_database_preprocessed"], encoding="utf-8")
# with pd.read_csv(config_front["known_clients_database_preprocessed"], encoding="utf-8", index_col="SK_ID_CURR",
#                chunksize=5000) as reader:
#  for i, data in enumerate(reader):
#     print(i, "_Update the list of chunks_ Shape : ", data.shape)
#    DATA_ALL_CLIENTS_CHUNKS.append(data)
#   gc.collect()
# gc.collect()

# income / age ?
# graphiques pour situer le client par rapport aux autres (features qui me paraissent)
# basé sur feature importance GLOBALE et comparer avec ses var
# basé sur shap ?

global CLIENT_ID
CLIENT_ID = 100005  # 456250

global CLIENT_JSON
global CLIENT_DF

# to save time from calling the endpoint global_feature_importance
global LIST_FEATURES

global NB_FEATURES_TO_PLOT
NB_FEATURES_TO_PLOT = 2

# we initialize the first webview to homepage
global DASHBOARD_CHOICE
DASHBOARD_CHOICE = "Homepage"

global PREDICTION


#################################################################"
# TODO refacto
def load_data():
    pass


# Web page #########################################################################

def initialize_webview():
    """
     Initialize webview with sidebar to choose client and which dashboard to display

    :param: None
    :return: None
    :rtype: None
    """
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
        global DASHBOARD_CHOICE
        DASHBOARD_CHOICE = st.radio('Menu', [
            'Homepage', 'Basic Dashboard', 'Advanced Dashboard',
            'Exploratory Data Analysis'
        ])  # label_visibility
        st.write('## ')
        st.write('## ')

        if DASHBOARD_CHOICE in ['Basic Dashboard', 'Advanced Dashboard']:
            # Client selector
            st.write('## Client ID:')

            global CLIENT_ID
            CLIENT_ID = st.number_input("Enter client ID", value=100005)
            st.caption("Example of client predicted negative (no default) : 100005")
            st.caption("Example of client predicted positive (credit default) : 456250")

            st.caption(" ")


        elif DASHBOARD_CHOICE == 'Exploratory Data Analysis':
            # st.write('## Choose data:')
            data_choice = st.radio('Choose data',
                                   ['Overview', 'bureau.csv', 'bureau_balance.csv', 'POS_CASH_balance.csv',
                                    'credit_card_balance.csv', 'previous_application.csv', 'installments_payments.csv',
                                    'application_train.csv'])  # , 'application_test.csv']) # label_visibility
            # data_choice = st.radio('Choose data', [
            #   'Overview', 'application_train.csv - part 1',
            #  'application_train.csv - part 2', 'application_train.csv - part 3',
            # 'bureau.csv', 'bureau_balance.csv', 'POS_CASH_balance.csv',
            # 'credit_card_balance.csv', 'previous_application.csv',
            # 'installments_payments.csv'
            # ])


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
    "This site contains an **interactive dashboard** to explain to the bank's customers the reason of **approval or refusal of their credit applications.**"
    "Probability of credit default has been calculated by a prediction model based on machine learning."
    " "
    " "
    "The bullet points of the prediction model are:"
    "- The data used for model training contain the entire set of tables available for the [Home Credit data repository at Kaggle.](https://www.kaggle.com/c/home-credit-default-risk/data)"
    "- The prediction model used to determine the probability of default of a credit application is based on the **LGBM algorithm** (Light Gradient Boosting Machine)."
    "- This model has been optimized with the intent to **minimize the buisness cost function** : each defaulting client costs 10 times the gain of a non-defaulting client."
    f"- The optimization  has lead to an **optimum threshold for the probability of default : {100 * THRESHOLD}%**. In other words, customer with a probability of default below {100 * THRESHOLD}% get their credit accepted, and refused if above {100 * THRESHOLD}%."
    " "
    " "
    "The dashboard is available in 2 versions:"
    "- A **basic** version, to be used by customer relation management."
    "- An **advanced**, more detailed version for deeper understanding of the data."
    "An **exploratory data analysis** is also available for the raw data used for the LGBM model training."
    " "


def basic_dashboard():
    """
    Basic dashboard with prediction and local feature importance

    :param: None
    :return: None
    :rtype: None
    """
    # load_data() # TODO correct
    global CLIENT_ID
    global CLIENT_JSON
    global CLIENT_DF

    # Main title of the dashboard
    st.title(f'Default Risk Prediction for client {CLIENT_ID}')

    # Get client data
    try:
        CLIENT_JSON = request_client_data(HOST + ENDPOINT_CLIENT_DATA, CLIENT_ID)
    except Exception as e:
        print("Exception raised while trying to get client data :\n\n", e)
        st.write('The client with the id {} is not in the database.'.format(CLIENT_ID))
        return  # to get out of the function

    proba_view()

    st.header("Global Feature Importance")
    global_feature_importance_view()
    shap_summary_plot_view()

    st.header("Local Feature Importance")
    shap_view()
    shap_force_plot_view()
    # TODO rename or Global VAR for client_df


def proba_view():
    """
     Result of credit application
    :param: None
    :return: None
    :rtype: None
    """
    st.header('Result of credit application')
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


def shap_view():
    """
    Local SHAP

    :param: None
    :return: None
    :rtype: None
    """
    st.header('Impact of features on prediction')
    try:
        # get shap values for the selected client
        client_shap_json = request_shap(HOST + ENDPOINT_SHAP, CLIENT_JSON)
        df_shap = json_to_df(client_shap_json)  # just need pd.Dataframe()
        shap_barplot(df_shap, DF_DESCRIPTION)
    except Exception as e:
        print("Exception raised :", e)


def advanced_dashboard():
    """
    The view will be the same as basic dashboard but we add more information about client compared to other clients in the database
    Position of the client vs other clients

    :param: None
    :return: None
    :rtype: None
    """
    global CLIENT_ID
    st.header('Ranking of the client compared to other clients')

    # INPUT HOWMANY FEATURE TO ANALYSED
    st.sidebar.write('______________________________')
    st.sidebar.write(' ')

    global NB_FEATURES_TO_PLOT
    NB_FEATURES_TO_PLOT = int(st.sidebar.selectbox(
        'Features range you want to analyse',
        [2, 3, 4, 5, 6]))

    st.sidebar.write('______________________________')
    st.sidebar.write(' ')

    # SLIDER FOR MOST IMPACTFUL FEATURES
    st.sidebar.write(
        f'*{str(NB_FEATURES_TO_PLOT)} most impactful features for selected client :* {LIST_FEATURES[:NB_FEATURES_TO_PLOT]}')

    # iterate over n MOST IMPACTFUL FEATURES
    boxplot_view()

    hist_view()

    # contourplot_view()


def shap_force_plot_view():  # TODO refacto client_df
    """
    Local SHAP

    :param: None
    :return: None
    :rtype: None
    """
    st.header('SHAP Force plot')
    try:
        # get shap values for the selected client
        response_shap = request_shap_expected(HOST + ENDPOINT_SHAP_EXPECTED, CLIENT_JSON)

        expected_value = response_shap["expected_value"]  # CF back end doc TODO refacto
        encodedNumpyData = response_shap["shap_values"]

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decodedArrays = json.loads(encodedNumpyData)
        finalNumpyArray = np.asarray(decodedArrays["array"])

        client_df = json_to_df(CLIENT_JSON)  # just need pd.Dataframe()

        shap_force_plot(finalNumpyArray, expected_value, PREDICTION, client_df)
    except Exception as e:
        print("Exception raised :", e)


def global_feature_importance_view():
    st.header('Global Feature Importance of the model used for the prediction')
    try:
        # get global feature importance
        response = request_feature_importance(HOST + ENDPOINT_FEATURE_IMPORTANCE, CLIENT_JSON)
        global_feature_importance_barplot(response, DF_DESCRIPTION)

        # here we keep the value ?? for the boxplot ??
        global LIST_FEATURES
        LIST_FEATURES = list(response.keys())  # the response is a dict {feature: value}

    except Exception as e:
        print("Exception raised :", e)


def shap_summary_plot_view():  # TODO refacto ??
    st.header('SHAP Feature Importance summary plot')
    try:
        # get shap values for the selected client
        response_shap = request_shap_expected(HOST + ENDPOINT_SHAP_EXPECTED, CLIENT_JSON)
        encodedNumpyData = response_shap["shap_values"]

        # Deserialization
        print("Decode JSON serialized NumPy array")
        decodedArrays = json.loads(encodedNumpyData)
        finalNumpyArray = np.asarray(decodedArrays["array"])
        shap_values = list(finalNumpyArray)  # TODO refacto / response is an array (2, 1, 777)
        # or shap_values should be a list of 2 arrays, not an array of two arrays

        client_df = json_to_df(CLIENT_JSON)  # just need pd.Dataframe()

        shap_summary_plot(shap_values, client_df)

    except Exception as e:
        print("Exception raised :", e)


#################################################################################""


def boxplot_view():
    # global NB_FEATURES_TO_PLOT  # so that the number of boxplot changes when chosen from sidebar
    # global LIST_FEATURES

    st.header(
        'Box plots for the most importance features using all the known clients and comparing to the current client')

    # We get the list of most importance features for the model
    # list_features = list(dict_f_i.keys())[:10]

    client_df = json_to_df(CLIENT_JSON)  # TODO add CLIENT_DF as global var

    print("__Reading database of all clients for the list of features : ", LIST_FEATURES[:NB_FEATURES_TO_PLOT])
    # we create a list to read the database / csv
    columns_list = LIST_FEATURES[:NB_FEATURES_TO_PLOT].copy()
    columns_list.extend(["SK_ID_CURR", "TARGET"])

    print("HEROKU CRASH ??????")

    # we read here the database using only the feature to plot and the index
    data_all_clients = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                                   usecols=columns_list)

    # for feature in LIST_FEATURES[:NB_FEATURES_TO_PLOT]:  # to display the number of graphs wanted
    boxplot_all_clients_compared_to_client_feature_value(data_all_clients, LIST_FEATURES[:NB_FEATURES_TO_PLOT],
                                                         client_df)


#########################################################################"

def hist_view():
    # global NB_FEATURES_TO_PLOT  # so that the number of boxplot changes when chosen from sidebar
    # global LIST_FEATURES

    st.header(
        'Histograms for the most importance features using all the known clients and comparing to the current client')

    # We get the list of most importance features for the model
    # list_features = list(dict_f_i.keys())[:10]

    client_df = json_to_df(CLIENT_JSON)  # TODO add CLIENT_DF as global var

    print("__Reading database of all clients for the list of features : ", LIST_FEATURES[:NB_FEATURES_TO_PLOT])
    # we create a list to read the database / csv
    columns_list = LIST_FEATURES[:NB_FEATURES_TO_PLOT].copy()
    columns_list.extend(["SK_ID_CURR", "TARGET"])

    FEATURE = "PAYMENT_RATE"

    print("HEROKU CRASH ??????")

    # we read here the database using only the feature to plot and the index
    data_all_clients = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                                   usecols=columns_list)

    # for feature in LIST_FEATURES[:NB_FEATURES_TO_PLOT]:  # to display the number of graphs wanted

    histgram_compared_to_all_clients(data_all_clients, FEATURE, client_df)


###########################################################################

def contourplot_view():
    st.header(
        'Bi variate analysis graph')

    feature_1 = 'AMT_ANNUITY'  # TODO remove and put sidebar
    feature_2 = 'EXT_SOURCE_2'  # TODO remove and put sidebar

    print("HEROKU CRASH ??????")
    # we read here the database using only the feature to plot and the index
    data_all_clients = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                                   usecols=[feature_1, feature_2, "SK_ID_CURR", "TARGET"])

    client_df = json_to_df(CLIENT_JSON)  # TODO add CLIENT_DF as global var

    contourplot(feature_1, feature_2, client_df, CLIENT_ID, data_all_clients, DF_DESCRIPTION)


##########################################################################################"


def eda_dashboard():
    """
    Exploratory Data Analysis graphs

    :param: None
    :return: None
    :rtype: None
    """
    ### TODO ###
    pass


#############################################################################################"

def dashboard_view():
    """
    Displays the correct view as a function of the dashboard selected in the menu.

    :param: None
    :return: None
    :rtype: None
    """
    # Homepage #######################################################
    if DASHBOARD_CHOICE == 'Homepage':
        homepage()

    # Basic and Advanced Dashboards #######################################################
    elif DASHBOARD_CHOICE in ['Basic Dashboard', 'Advanced Dashboard']:
        basic_dashboard()

    if DASHBOARD_CHOICE == 'Advanced Dashboard':
        advanced_dashboard()

    else:  # EDA graphs
        eda_dashboard()


# analyse sur le client (stat) : sur les 3 var les plus importantes // comaprer avec la moy des clients refusés et acceptes /
# application_train : SUR UNE VAR moy des clients 0 / moy des clients 1 // place mon client par rapport à eux (revenus)
# Feature importance SHAP globale change pas (meme graph pour chaque client)
# feature important SHAP local (à la fin du notebook de prediction)
# sur DB : ajouter info sur client / genre - salaire - etc

# salaire : si modifie / personne acceptée ou pas (bonus)

######################################################################

def main():
    # Sidebar
    initialize_webview()

    # choose dashboard type
    dashboard_view()


if __name__ == '__main__':
    main()
