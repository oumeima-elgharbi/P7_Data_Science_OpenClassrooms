import yaml
import joblib


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


def load_model(model_file):
    """

    :param model_file:
    :return:
    """
    model = joblib.load(model_file)
    return model


def get_prediction_proba(model, client_df):
    """

    :param model:
    :param client_df:
    :return:
    """
    probability = model.predict_proba(client_df)
    return probability
