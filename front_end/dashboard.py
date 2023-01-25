# to find the parent direcory

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

import requests

from functions_dashboard import *
from functions import *

# SHAP : featu importance globale (constt) // local : le client 4 : telle var plus impacte sur son score et diff de feat importance
# global : 3e dans la lsite mais si client X : 1er revenu

# SMOTE : classes desequilibrées : dummy 0 : pour améliorer score sur classe 1 et réequilibrer dataset
# voir si ca améliore le score


# 0) Variables
THRESHOLD = 0.4

# 1) Endpoints declaration

HOST = 'http://127.0.0.1:8000'

endpoint_get_client_data = "/clients/{}/"
endpoint_predict = "/predict/"
endpoint_shap = "/shap/"

client_id = st.sidebar.number_input('Insert client id', value=100001)  # default value 100001 # to change later to 0


# 2) GET client / POST predict / POST shap

def request_client_data(model_uri, client_id):
    """

    :param model_uri:
    :param client_id: (int)
    :return:
    """
    response = requests.request(
        method='GET', url=model_uri.format(client_id))

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def request_prediction(model_uri, client_json):
    """

    :param model_uri:
    :param client_json:
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=client_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def request_shap(model_uri, client_json):
    """

    :param model_uri:
    :param client_json:
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=client_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


# 3) dashboard front-end

def main():
    st.title("Home Credit Default Risk Prediction")
    st.title('Client n°{} application for a loan'.format(client_id))

    # Get client data
    # we get the json body for the client_id selected
    client_json = request_client_data(HOST + endpoint_get_client_data, client_id)

    # get shap values for the selected client
    client_shap_json = request_shap(HOST + endpoint_shap, client_json)


    # Local SHAP
    ### ?? ### "---------------------------"
    st.header('Impact of features on prediction')
    df_shap = json_to_df(client_shap_json) # just need pd.Dataframe()
    ##st.write(df_shap.shape) ### for test purposes
    shap_barplot(df_shap)


    predict_btn = st.button('Prédire')
    if predict_btn:
        proba = None  # ??
        pred = None

        proba = request_prediction(HOST + endpoint_predict, client_json) # we get the prediction

        if proba <= THRESHOLD:
            pred = 0
            result = "yes"
        else:
            pred = 1
            result = "no"

        st.write('Probability that the loan is not payed back {} %'.format(round(100 * proba, 2)))
        st.write('Loan accepted : {}'.format(result))

        ############################################################
        # Gauge
        st.header('Gauge prediction')
        rectangle_gauge(client_id, proba)
        ###################################""





if __name__ == '__main__':
    main()
    # streamlit run dashboard.py
