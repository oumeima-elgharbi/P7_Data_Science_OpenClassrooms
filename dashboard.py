import pandas as pd
import streamlit as st
import requests

HOST = 'http://127.0.0.1:8000'

client_id = st.sidebar.number_input('Insert client id',
                                    value=100001)


# payload = {
#    "client_id": str(client_id)
# }

def request_prediction(model_uri, client_id):
    # headers = {"Content-Type": "application/json"}

    # data_json = {'data': data}
    response = requests.request(
        method='POST', url=model_uri + str(client_id))  # headers=headers, url=model_uri)#, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    FASTAPI_URI = HOST + '/v1/prediction/client/'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['FastAPI'])

    st.title('Client n° application for a loan')

    revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
                                 min_value=0., value=3.87, step=1.)

    predict_btn = st.button('Prédire')
    if predict_btn:
        pred = None

        if api_choice == 'FastAPI':
            pred = request_prediction(FASTAPI_URI, client_id)

        st.write(
            'Response {}'.format(pred))


if __name__ == '__main__':
    main()
    # streamlit run dashboard.py
