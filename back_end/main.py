# to find the parent directory

import sys
import os
#import gc

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
#import uvicorn

#from utils import *
from back_end.prediction_functions import * # TODO remove
from back_end.preprocessing import * # TODO remove

# before opening the web service, we load all the models and files
print("__Getting config")
config = read_yml("config.yml")

print("__Deployment : {}__".format(config["deploy"]))
if config["deploy"]:
    HOST = 'https://p7-data-science-openclassrooms.herokuapp.com/'
else:
    HOST = 'http://127.0.0.1:8000'

print("__Getting config back-end__")
config_back = read_yml("back_end/config_backend.yml")

print("__Unzip model and dataset__")
unzip_file(path_to_zip_file=config_back["resources"]["zip"], directory_to_extract_to=config_back["resources"]["unzip"])

print("__Loading classifier__")
model = load_model(config_back["classifier"])

print("__Loading database of preprocessed clients__")
data_all_clients = pd.read_csv(config_back["clients_database_preprocessed"])

# Create a FastAPI instance
app = FastAPI()

@app.get('/')
async def index():
    """
    Welcome message.
    Args:
    - None.
    Returns:
    - Message (string).
    """
    return 'Hello, you are accessing an API'


@app.post('/client/{client_id}/')
async def get_client_data(client_id: int):
    """
    Body empty, using the client's id, we get the client's preprocessed data

    :param client_id: (int)
    :return: a preprocessed client with feature / value
    :rtype: (dict)
    """
    print("__Getting client's application data from database__")
    client_df = preprocess_one_application(data_all_clients, client_id)
    print("HERE4")
    client_json = df_to_json(client_df)
    print("HERE5")
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
async def get_shap(client_json: dict = Body({})):
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

#if __name__ == '__main__':
#    # opening the web service
#    uvicorn.run(app,
#                host=HOST.split(":")[0],
#                port=HOST.split(":")[1])
#ipython