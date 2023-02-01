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

# 0) Config : unzip data and check host if deployment or not

print("_____Getting config_____")
config = read_yml("config.yml")

#print("__Unzip model and dataset__")
#unzip_file(path_to_zip_file=config["resources"]["zip"], directory_to_extract_to=config["resources"]["unzip"])

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

# TODO add dtype reading !!! https://stackoverflow.com/questions/50047237/how-to-preserve-dtypes-of-dataframes-when-using-to-csv
DATA_ALL_CLIENTS = pd.read_csv(config_front["known_clients_database_preprocessed"], encoding="utf-8")

# gc.collect()

global CLIENT_ID
CLIENT_ID = 100001  # 456250

global CLIENT_JSON

# we initialize the first webview to homepage
global DASHBOARD_CHOICE
DASHBOARD_CHOICE = "Homepage"

global PREDICTION


#################################################################"
# TODO refacto
def load_data():
    # Load data
    list_categorical_features = load_data('list_categorical_features')
    dict_categorical_features = load_data('dict_categorical_features')
    list_quantitative_features = load_data('list_quantitative_features')
    list_features_permutation_importances = load_data('list_features_permutation_importances')
    list_summary_plot_shap = load_data('list_summary_plot_shap')


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
        logo_homecredit = Image.open('./img/Home-Credit-logo.jpg')
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
            CLIENT_ID = st.number_input("Enter client ID", value=100001)
            st.caption("Example of client predicted negative (no default) : 100001")
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
    shap_view()


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

    ### TODO ###


def lineplot_view(feature=""):  ## TODO working on this
    """

    """
    # we create a df for the client in json format and add it to the df of all clients
    df_client = json_to_df(CLIENT_JSON)
    data_all_clients = add_new_client_to_data_all_clients(DATA_ALL_CLIENTS, df_client, PREDICTION)

    # TODO FINISH function
    lineplot(DATA_ALL_CLIENTS, df_client, CLIENT_ID, THRESHOLD, feature, DF_DESCRIPTION)


def eda_dashboard():
    """
    Exploratory Data Analysis graphs

    :param: None
    :return: None
    :rtype: None
    """
    ### TODO ###
    pass


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
