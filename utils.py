import pandas as pd
import json
import yaml


def read_yml(file):
    """
    Function that reads a yaml file and loads its content into a json format ?
    :param file: (file) the yaml file that we want to read
    :return: The yaml file content
    :rtype: dict ?

    :UC: The file must be a yaml file.
    """
    assert file.endswith('.yaml') or file.endswith('.yml'), "The file given is not a yaml/yml file."
    with open(file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def json_to_df(client_json):
    """

    :param client_json:
    :return: (DataFrame)
    """
    if type(client_json) == dict:  # one dict / for clients
        client_df = pd.Series(client_json).to_frame().transpose()
        # client_df = pd.DataFrame.from_dict(client_json, orient="index").transpose()
    elif type(client_json) == list:  # records / list of dict
        client_df = pd.DataFrame(client_json)  ## for shap
    return client_df


def df_to_json(client_df):
    """

    :param client_df:
    :return: (list) list of dict
    """
    client_string = client_df.to_json(orient="records")
    client_json = json.loads(client_string)  # list of dict
    # json.dumps(client_json, indent=4)
    return client_json
