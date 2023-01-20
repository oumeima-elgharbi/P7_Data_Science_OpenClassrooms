# Library imports
from fastapi import FastAPI, Body

from prediction_functions import load_model, get_prediction_proba, read_yml
from preprocessing import preprocess_one_application

HOST = 'http://127.0.0.1:8000'
# HOST = 'https://project7-api-ml.herokuapp.com'

# Create a FastAPI instance
app = FastAPI()

# at the opening of the web service, we load all the models and files

print("__Getting config")
config = read_yml("config.yml")

print("__Loading classifier")
model = load_model(config["classifier"])


# Endpoint

@app.post("/v1/prediction/client/{client_id}")
async def predict(client_id: int):  # remove async def ?? # :dict = Body({})
    """

    :param client_id:
    :return:
    """
    print("_____Start of prediction pipeline_____")
    client_preprocessed = preprocess_one_application(client_id)

    print("_____Predicting_____")
    probability = get_prediction_proba(model, client_preprocessed)
    # print(probability)
    # print(probability[:, 1])
    proba = round(probability[:, 1][0] * 100, 2)
    response = [{'client ID': client_id}, {'probability of paying back the loan in %': proba}]
    return response  # return prediction / [0] to get the value
