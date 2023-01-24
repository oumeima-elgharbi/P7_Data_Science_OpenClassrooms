# Library imports
from fastapi import FastAPI, Body

from prediction_functions import load_model, get_prediction_proba, read_yml
from preprocessing import preprocess_one_application
import pandas as pd

import shap
import json

HOST = 'http://127.0.0.1:8000'
# HOST = 'https://project7-api-ml.herokuapp.com'
# HEROKU_HOST = 'https://scoring-credit-oc-48975.herokuapp.com'

# Create a FastAPI instance
app = FastAPI()

# at the opening of the web service, we load all the models and files

print("__Getting config")
config = read_yml("config.yml")

print("__Loading classifier")
model = load_model(config["classifier"])


# @app.get('/clients/{client_id}')
# def base(client_id: int):
#    return f"Client id is {client_id} & it's type is : {type(client_id)}"

@app.get('/clients/{client_id}')
def get_client_data(client_id: int):
    """
    Body empty, using the client's id, we get the client's preprocessed data
    
    :param client_id:
    :param real_time: if False, we return the preprocessed client's application
    :return:
    """
    print("__Getting client's application data from database__")
    client_df = preprocess_one_application(client_id)

    client_string = client_df.to_json(orient="records")
    client_json = json.loads(client_string)  # list of dict
    # json.dumps(client_json, indent=4)
    return client_json[0]


@app.post("/predict/")
async def predict(client_json: dict = Body({})):  # remove async def ?? # :dict = Body({})
    """
    Put the preprocessed data of the client in the Body
    Returns the probability that the client will repay the loan

    :param client_json: (dict) json formatted preprocessed client
    :return:
    """
    print("_____Start of prediction pipeline_____")
    print("_____Getting client_____")
    client_df = pd.DataFrame.from_dict(client_json, orient="index").transpose()
    # client_df = pd.Series(client_json).to_frame().transpose()

    print("_____Predicting_____")
    probability = get_prediction_proba(model, client_df)
    # print(probability)
    # print(probability[:, 1])
    # proba = round(probability[:, 1][0] * 100, 2)
    # response = [{'client ID': client_id}, {'probability of paying back the loan in %': proba}]
    response = probability[:, 1][0]
    return response  # return prediction / [0] to get the value


# analyse sur le client (stat) : sur les 3 var les plus importantes // comaprer avec la moy des clients refusés et acceptes /
# application_train : SUR UNE VAR moy des clients 0 / moy des clients 1 // place mon client par rapport à eux (revenus)
# barre jauge proba / (PM)
# Feature importance SHAP globale change pas (meme graph pour chaque client)
# feature important SHAP local (à la fin du notebook de prediction)
# Streamlit / SHAP
# sur DB : ajouter info sur client / genre - salaire - etc

# salaire : si modifie / personne acceptée ou pas (bonus)

#######################
@app.get('/shap/')
def get_shap(json_client: dict = Body({})):
    """
    Calculates the probability of default for a client.
    Args:
    - client data (json).
    Returns:
    - SHAP values (json).
    """
    df_one_client = pd.Series(json_client).to_frame().transpose()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_one_client)
    df_shap = pd.DataFrame({
        'SHAP value': shap_values[1][0],
        'feature': df_one_client.columns
    })
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)
    return df_shap.to_json(orient='index')
#############################""
