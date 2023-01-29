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

print("__Unzip model and dataset__")
unzip_file(path_to_zip_file=config["resources"]["zip"], directory_to_extract_to=config["resources"]["unzip"])

print("Deployment ? {}".format(config["deploy"]["is"]))
if config["deploy"]["is"]:
    HOST = config["deploy"]["prod_api"]
else:
    HOST = config["deploy"]["dev"]

# 1) Config front-end
print("_____Getting config front-end_____")
config_front = read_yml("config_frontend.yml")

THRESHOLD = config_front["threshold"]
ENDPOINT_GET_CLIENT_DATA = config_front["endpoints"]["endpoint_get_client_data"]
ENDPOINT_PREDICT = config_front["endpoints"]["endpoint_predict"]
ENDPOINT_SHAP = config_front["endpoints"]["endpoint_shap"]
ENDPOINT_CLIENT_DATA = config_front["endpoints"]["endpoint_client_data"]

DF_DESCRIPTION = pd.read_csv(config_front["columns_description"], encoding="ISO-8859-1")  # not encoded in utf-8

gc.collect()


##################################################################################################################


# 456250

# 2) POST client / POST predict / POST shap

#################################################################"
# TODO refacto
def load_data():
    # Load data
    # dataset used to plot features - sample of clients from the Kaggle train set
    X_split_valid = load_data('X_split_valid', path='./src/')
    # dataset used to plot features - sample of clients from the Kaggle train set
    y_split_valid = load_data('y_split_valid', path='./src/')

    list_categorical_features = load_data('list_categorical_features')
    dict_categorical_features = load_data('dict_categorical_features')
    list_quantitative_features = load_data('list_quantitative_features')

    list_features_permutation_importances = load_data(
        'list_features_permutation_importances')
    list_summary_plot_shap = load_data('list_summary_plot_shap')


# Web page #########################################################################

def initialize_webview():
    # Load data
    # Default settings. This must be the first Streamlit command used in your app, and must only be set once.
    st.set_page_config(page_title="Project 7 Dashboard",
                       initial_sidebar_state="expanded",
                       layout="wide")
    # st.set_page_config(layout="wide")  ## remove ??

    # Side bar
    with st.sidebar:
        image_HC = Image.open('./img/Home-Credit-logo.png')
        st.image(image_HC, width=300)

        # Dashboard selector
        st.write('## Site Map:')
        dashboard_choice = st.radio('', [
            'Homepage', 'Basic Dashboard', 'Advanced Dashboard',
            'Exploratory Data Analysis'
        ])
        st.write('## ')
        st.write('## ')

        if dashboard_choice in ['Basic Dashboard', 'Advanced Dashboard']:
            # Client selector
            st.write('## Client ID:')
            global CLIENT_ID
            # CLIENT_ID = st.sidebar.number_input('Insert client id', value=456250)  # default value 100001 # to change later to 0
            CLIENT_ID = st.text_input("Enter client ID", value=100001)
            # st.caption("Example of client predicted negative (no default) : 324806")
            # st.caption("Example of client predicted positive (credit default) : 318063")
            st.caption(" ")

            # Button random
            # if st.button("Select random client"):
            #   clients = [str(id) for id in df_test_sample.index]
            #  size = len(clients)
            # client_index = randint(0, size - 1)
            # CLIENT_ID = clients[client_index]

        elif dashboard_choice == 'Exploratory Data Analysis':
            st.write('## Choose data:')
            # data_choice = st.radio('', ['Overview', 'bureau.csv', 'bureau_balance.csv', 'POS_CASH_balance.csv', 'credit_card_balance.csv', 'previous_application.csv', 'installments_payments.csv', 'application_train.csv', 'application_test.csv'])
            data_choice = st.radio('', [
                'Overview', 'application_train.csv - part 1',
                'application_train.csv - part 2', 'application_train.csv - part 3',
                'bureau.csv', 'bureau_balance.csv', 'POS_CASH_balance.csv',
                'credit_card_balance.csv', 'previous_application.csv',
                'installments_payments.csv'
            ])


###################################################################################
###################################################################################

def homepage():
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
    load_data()

    # Main title of the dashboard
    st.title(f'Default Risk Prediction for client {CLIENT_ID}')

    # Get client data
    try:
        global CLIENT_JSON
        CLIENT_JSON = request_client_data(HOST + ENDPOINT_CLIENT_DATA, CLIENT_ID)
    except Exception as e:
        print("Exception raised while trying to get client data :\n\n", e)
        st.write('The client with the id {} is not in the database.'.format(CLIENT_ID))

    # Result of credit application
    "---------------------------"
    st.header('Result of credit application')

    try:
        probability = request_prediction(HOST + ENDPOINT_PREDICT, CLIENT_JSON)
        if probability < THRESHOLD:
            st.success(
                f"  \n __CREDIT ACCEPTED__  \n  \nThe probability of default of the applied credit is __{round(100 * probability, 1)}__% (lower than the threshold of {100 * THRESHOLD}% for obtaining the credit).  \n "
            )
        else:
            st.error(
                f"__CREDIT REFUSED__  \nThe probability of default of the applied credit is __{round(100 * probability, 1)}__% (higher than the threshold of {100 * THRESHOLD}% for obtaining the credit).  \n "
            )
        rectangle_gauge(CLIENT_ID, probability, THRESHOLD)
    except Exception as e:
        print("Exception raised :", e)


def advanced_dashboard():
    # Position of the client vs other clients
    "---------------------------"
    st.header('Ranking of the client compared to other clients')

    ### TODO ###


def dashboard_view(dashboard_choice):
    # Homepage #######################################################
    if dashboard_choice == 'Homepage':
        homepage()

    # Basic and Advanced Dashboards #######################################################
    elif dashboard_choice in ['Basic Dashboard', 'Advanced Dashboard']:
        basic_dashboard()

    if dashboard_choice == 'Advanced Dashboard':
        advanced_dashboard()


####################################################### MAIN

def main():
    # side bar
    initialize_webview()

    dashboard_view(dashboard_choice="Homepage")


#####################################################

# 3) dashboard front-end
def main_old():
    ####################################################
    global CLIENT_ID
    st.title("Home Credit Default Risk Prediction")
    CLIENT_ID = st.sidebar.number_input('Insert client id', value=456250)  # default value 100001 # to change later to 0
    st.title('Client n°{} application for a loan'.format(CLIENT_ID))

    dashboard_view(dashboard_choice="Homepage")

    # gc.collect()
    ################################################################"""
    try:
        CLIENT_JSON = request_client_data(HOST + ENDPOINT_CLIENT_DATA, CLIENT_ID)
    except Exception as e:
        print("Exception raised while trying to get client data :\n\n", e)
        st.write('The client with the id {} is not in the database.'.format(CLIENT_ID))

    # Local SHAP
    #####################################################
    st.header('Impact of features on prediction')
    try:
        # get shap values for the selected client
        client_shap_json = request_shap(HOST + ENDPOINT_SHAP, CLIENT_JSON)
        df_shap = json_to_df(client_shap_json)  # just need pd.Dataframe()
        ##st.write(df_shap.shape) ### for test purposes
        shap_barplot(df_shap, DF_DESCRIPTION)

    except Exception as e:
        print("Exception raised :", e)

    ###################################################################################
    predict_btn = st.button('Prédire')
    if predict_btn:
        proba = 0  # ??
        pred = -1

        try:
            proba = request_prediction(HOST + ENDPOINT_PREDICT, CLIENT_JSON)  # we get the prediction
            if proba <= THRESHOLD:
                pred = 0
                result = "yes"
            else:
                pred = 1
                result = "no"
            st.write('Probability that the loan is not payed back {} %'.format(round(100 * proba, 2)))
            st.write('Loan accepted : {}'.format(result))
        except Exception as e:
            print("Exception raised :", e)

        ############################################################
        # Gauge
        st.header('Gauge prediction')
        rectangle_gauge(CLIENT_ID, proba, THRESHOLD)
        ###################################""


############################################################"

if __name__ == '__main__':
    main()
    # streamlit run dashboard.py OR streamlit run front_end/dashboard.py
