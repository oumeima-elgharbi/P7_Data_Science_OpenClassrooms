import pandas as pd
import streamlit as st
import requests

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
    st.title('Client n°{} application for a loan'.format(client_id))

    predict_btn = st.button('Prédire')
    if predict_btn:
        pred = None  # ??

        # we get the json body for the client_id selected
        client_json = request_client_data(HOST + endpoint_get_client_data, client_id)
        pred = request_prediction(HOST + endpoint_predict, client_json) # we get the prediction

        st.write('Probability that the client pays back the loan {} %'.format(round(100 * pred, 2)))


if __name__ == '__main__':
    main()
    # streamlit run dashboard.py
