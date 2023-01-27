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

import requests
from functions_dashboard import *
from utils import *

# SHAP : featu importance globale (constt) // local : le client 4 : telle var plus impacte sur son score et diff de feat importance
# global : 3e dans la lsite mais si client X : 1er revenu

# SMOTE : classes desequilibrées : dummy 0 : pour améliorer score sur classe 1 et réequilibrer dataset
# voir si ca améliore le score

# 0) Config : unzip data and check host if deployment or not

print("_____Getting config_____")
config = read_yml("config.yml")

print("__Unzip model and dataset__")
unzip_file(path_to_zip_file=config["resources"]["zip"], directory_to_extract_to=config["resources"]["unzip"])

print("Deployment ? {}".format(config["deploy"]))
if config["deploy"]:
    HOST = 'https://p7-data-science-openclassrooms.herokuapp.com/'
else:
    HOST = 'http://127.0.0.1:8000'

# 1) Config front-end
print("_____Getting config front-end_____")
config_front = read_yml("front_end/config_frontend.yml")

THRESHOLD = config_front["threshold"]
ENDPOINT_GET_CLIENT_DATA = config_front["endpoints"]["endpoint_get_client_data"]
ENDPOINT_PREDICT = config_front["endpoints"]["endpoint_predict"]
ENDPOINT_SHAP = config_front["endpoints"]["endpoint_shap"]
DF_DESCRIPTION = pd.read_csv(config_front["columns_description"], encoding="ISO-8859-1")  # not encoded in utf-8

CLIENT_ID = st.sidebar.number_input('Insert client id', value=456250)  # default value 100001 # to change later to 0


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
    st.title('Client n°{} application for a loan'.format(CLIENT_ID))

    # Get client data
    # we get the json body for the client_id selected
    client_json = request_client_data(HOST + ENDPOINT_GET_CLIENT_DATA, CLIENT_ID)

    # get shap values for the selected client
    client_shap_json = request_shap(HOST + ENDPOINT_SHAP, client_json)

    # Local SHAP
    ### ?? ### "---------------------------"
    st.header('Impact of features on prediction')
    df_shap = json_to_df(client_shap_json)  # just need pd.Dataframe()
    ##st.write(df_shap.shape) ### for test purposes
    shap_barplot(df_shap, DF_DESCRIPTION)

    predict_btn = st.button('Prédire')
    if predict_btn:
        proba = None  # ??
        pred = None

        proba = request_prediction(HOST + ENDPOINT_PREDICT, client_json)  # we get the prediction

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
        rectangle_gauge(CLIENT_ID, proba, THRESHOLD)
        ###################################""


if __name__ == '__main__':
    main()
    # streamlit run dashboard.py OR streamlit run front_end/dashboard.py
