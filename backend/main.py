# Library imports
from fastapi import FastAPI, Body
import uvicorn

from prediction_functions import *  # TODO remove
from preprocessing import *  # TODO remove

# before opening the web service, we load all the models and files
print("_____Getting config_____")
config = read_yml("config.yml")

print("__Unzip model and dataset__")
unzip_file(path_to_zip_file=config["resources"]["zip"], directory_to_extract_to=config["resources"]["unzip"])

print("__Deployment : {}__".format(config["deploy"]["is"]))
if config["deploy"]["is"]:
    HOST = config["deploy"]["prod_api"]
else:
    HOST = config["deploy"]["dev"]

print("_____Getting config back-end_____")
config_back = read_yml("config_backend.yml")

print("__Loading classifier__")
model = load_model(config_back["classifier"])

gc.collect()

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


@app.post('/client_data')  # 3s
async def client_data(client: dict = Body({})):
    """
    Body empty, using the client's id, we get the client's preprocessed data

    :return: a preprocessed client with feature / value
    :rtype: (dict)
    """
    print("__Getting client's application data from database__")
    # {"client_id": 0}
    try:
        client_df = preprocess_one_application(client["client_id"])
        client_json = df_to_json(client_df)
        return client_json[0]
    except Exception as e:
        print(e)
        print("Couldn't get client data from database.")


@app.post("/predict")
async def predict(client_json: dict = Body({})):  # remove async def ?? # :dict = Body({})
    """
    Put the preprocessed data of the client in the Body
    Returns the probability that the client will repay the loan

    :param client_json: (dict) json formatted preprocessed client
    :return: the probability that the client will repay the loan
    :rtype: (float)
    """
    assert client_json != {}, "There is no data for the client"  # TODO add pydantics verification
    print("_____Start of prediction pipeline_____")
    print("_____Getting client_____")
    client_df = json_to_df(client_json)

    print("_____Predicting_____")
    probability = get_prediction_proba(model, client_df)
    response = probability[:, 1][0]  # return prediction / [0] to get the value
    return response


@app.post('/shap')
async def get_shap(client_json: dict = Body({})):
    """
    Computes SHAP values for each feature for a client
    ##the probability of default for a client.

    :param client_json: (dict) json formatted preprocessed client
    :return: List of dict, each dict is a feature + its SHAP value
    :rtype: (list)
    """
    assert client_json != {}, "There is no data for the client"  # TODO add pydantics verification
    print("_____Start of SHAP_____")
    client_df = json_to_df(client_json)

    print("_____Getting SHAP for our client_____")
    df_shap = get_shap_values(model, client_df)

    # transforming df to json response
    client_shap_json = df_to_json(df_shap)

    print("We can verify that we have as much SHAP values as we have features for our client : ", len(client_shap_json),
          len(client_shap_json) == client_df.shape[1])
    return client_shap_json


if __name__ == '__main__':
    # opening the web service
    uvicorn.run(app,
                host=HOST.split(":")[0],
                port=HOST.split(":")[1])
