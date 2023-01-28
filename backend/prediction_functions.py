# to find the parent directory
import sys
import os

# getting the name of the directory
# where the file is present.
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

import joblib
import shap

from utils import *


def load_model(model_file):
    """

    :param model_file: (string)
    :return:
    :rtype:
    """
    model = joblib.load(model_file)
    return model


def get_prediction_proba(model, client_df):
    """

    :param model: serialized model that has a predict_proba method
    :param client_df:
    :return:
    """
    probability = model.predict_proba(client_df)
    return probability


def get_shap_values(model, client_df):
    """

    :param model: serialized model that has a predict_proba method
    :param client_df: (DataFrame)
    :return:
    :rtype: (DataFrame)
    """

    explainer = shap.TreeExplainer(model) # TODO : LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
    shap_values = explainer.shap_values(client_df)
    df_shap = pd.DataFrame({
        'SHAP value': shap_values[1][0],
        'feature': client_df.columns
    })
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)

    return df_shap
