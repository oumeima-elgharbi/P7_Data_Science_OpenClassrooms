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

# Library imports
from fastapi import FastAPI, Body

from prediction_functions import *
from preprocessing import *
from utils import *

# Create a FastAPI instance
app = FastAPI()

# at the opening of the web service, we load all the models and files

print("__Getting config")
config = read_yml("../config.yml")

print("Deploiment ? {}".format(config["deploy"]))
if config["deploy"]:
    HOST = 'https://p7-data-science-openclassrooms.herokuapp.com/'
else:
    HOST = 'http://127.0.0.1:8000'

print("__Getting config back-end")
config_back = read_yml("config_backend.yml")

print("__Loading classifier")
model = load_model(config_back["classifier"])


@app.get('/clients/{client_id}')
def get_client_data(client_id: int):
    """
    Body empty, using the client's id, we get the client's preprocessed data

    :param client_id: (int)
    :return: a preprocessed client with feature / value
    :rtype: (dict)
    """
    print("__Getting client's application data from database__")
    client_df = preprocess_one_application(client_id)
    client_json = df_to_json(client_df)
    return client_json[0]


@app.post("/predict/")
async def predict(client_json: dict = Body({})):  # remove async def ?? # :dict = Body({})
    """
    Put the preprocessed data of the client in the Body
    Returns the probability that the client will repay the loan

    :param client_json: (dict) json formatted preprocessed client
    :return: the probability that the client will repay the loan
    :rtype: (float)
    """
    print("_____Start of prediction pipeline_____")
    print("_____Getting client_____")
    client_df = json_to_df(client_json)

    print("_____Predicting_____")
    probability = get_prediction_proba(model, client_df)
    response = probability[:, 1][0]  # return prediction / [0] to get the value
    return response


@app.post('/shap/')
def get_shap(client_json: dict = Body({})):
    """
    Computes SHAP values for each feature for a client
    ##the probability of default for a client.

    :param client_json: (dict) json formatted preprocessed client
    :return: List of dict, each dict is a feature + its SHAP value
    :rtype: (list)
    """
    print("_____Start of SHAP_____")
    client_df = json_to_df(client_json)

    print("_____Getting SHAP for our client_____")
    df_shap = get_shap_values(model, client_df)

    # transforming df to json response
    client_shap_json = df_to_json(df_shap)

    print("We can verify that we have as much SHAP values as we have features for our client : ", len(client_shap_json),
          len(client_shap_json) == client_df.shape[1])
    return client_shap_json

# analyse sur le client (stat) : sur les 3 var les plus importantes // comaprer avec la moy des clients refusés et acceptes /
# application_train : SUR UNE VAR moy des clients 0 / moy des clients 1 // place mon client par rapport à eux (revenus)
# barre jauge proba / (PM)
# Feature importance SHAP globale change pas (meme graph pour chaque client)
# feature important SHAP local (à la fin du notebook de prediction)
# Streamlit / SHAP
# sur DB : ajouter info sur client / genre - salaire - etc

# salaire : si modifie / personne acceptée ou pas (bonus)
