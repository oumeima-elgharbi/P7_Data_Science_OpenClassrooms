# Library imports
from fastapi import FastAPI, Body
import uvicorn

from prediction_functions import *  # TODO remove
from preprocessing import *  # TODO remove

from json import JSONEncoder
import numpy as np

#############################################################################
# TODO remove and download data at the startup of the app

import os
from os import listdir

this_dir = os.getcwd()
all_files = [f for f in listdir(this_dir)]

if "resources" not in all_files:
    import subprocess

    print("__Download resources folder__")
    subprocess.call(r'python script_download_data_folder.py', shell=True)


##############################################################################

# to send a numpy array as a json response
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# before opening the web service, we load all the models and files
print("_____Getting config_____")
config = read_yml("config.yml")

# print("__Unzip model and dataset__")
# unzip_file(path_to_zip_file=config["resources"]["zip"], directory_to_extract_to=config["resources"]["unzip"])

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
    Welcome message to check that the API is working
    :param: None
    :return: a message that says that this is an API
    :rtype: (string)
    """
    return 'Hello, you are accessing an API'


@app.post('/client_data')  # 3s
async def client_data(client: dict = Body({})):
    """
    Body empty, using the client's id, we get the client's preprocessed data

    Body : {"client_id": 100001, "is_new_client": True} TODO revoir

    :return: a preprocessed client with feature / value
    :rtype: (dict)
    """
    print("__Getting client's application data from database__")
    # {"client_id": 0}
    try:
        client_df = preprocess_one_application(
            client["client_id"])  # TODO add option for database name / add in fct preprocess in the body !!
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
async def shap_values(client_json: dict = Body({})):
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
    shap_value = get_shap_values(model, client_df)
    df_shap = get_df_shap(shap_value, client_df)

    # transforming df to json response
    client_shap_json = df_to_json(df_shap)

    print("We can verify that we have as much SHAP values as we have features for our client : ", len(client_shap_json),
          len(client_shap_json) == client_df.shape[1])
    return client_shap_json


@app.post('/shap_expected')  # TODO refacto
async def shap_expected_values(client_json: dict = Body({})):
    """
    Computes SHAP values for each feature for a client
    ##the probability of default for a client.

    :return: List of dict, each dict is a feature + its SHAP value
    :rtype: (list)
    """
    assert client_json != {}, "There is no data for the client"  # TODO add pydantics verification
    print("_____Start of SHAP_____")
    client_df = json_to_df(client_json)

    print("_____Getting SHAP expected values from model_____")
    shap_value, shap_expected_values_list = get_shap_expected_value(model, client_df)

    numpy_data = {"array": shap_value}
    encoded_numpy_data = json.dumps(numpy_data, cls=NumpyArrayEncoder)  # use dump() to write array into file

    # shap_expected_values_list is a list because we called .shap_values
    response = {"expected_value": shap_expected_values_list,
                "shap_values": encoded_numpy_data}

    return response


@app.post('/feature_importance')
async def get_global_feature_importance(client_json: dict = Body({})):
    """
    Computes Global Feature Importance of the model
    TODO refacto : we are now using the client's data to get the columns names

    :param client_json: (dict) json formatted preprocessed client
    :return:
    :rtype: (dict)
    """
    assert client_json != {}, "There is no data for the client"  # TODO add pydantics verification
    print("_____Getting Global Feature Importance_____")
    client_df = json_to_df(client_json)  # TODO refactor remove client_df from parameters

    columns = client_df.columns.tolist()  # need this to serialize into json object
    global_feature_importance = model.feature_importances_.tolist()  # need this to serialize into json object

    # create dict {columns_name:model_feature_importance}
    dict_f_i = dict(zip(columns, global_feature_importance))  # model.feature_importances_
    # sorted by feature_importance
    dict_f_i = {k: v for k, v in sorted(dict_f_i.items(), key=lambda item: item[1], reverse=True)}

    return dict_f_i


if __name__ == '__main__':
    # opening the web service
    uvicorn.run(app,
                host=HOST.split(":")[0],
                port=HOST.split(":")[1])
