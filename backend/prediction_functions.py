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


'''
def get_shap_values(model, client_df):
    """

    :param model: serialized model that has a predict_proba method
    :param client_df: (DataFrame)
    :return:
    :rtype: (DataFrame)
    """

    explainer = shap.TreeExplainer(
        model)  # TODO : LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
    shap_values = explainer.shap_values(client_df)
    # return shap_values, explainer
    df_shap = pd.DataFrame({
        'SHAP value': shap_values[1][0],
        'feature': client_df.columns
    })
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)

    return df_shap

'''


#################""


def get_shap_values(model, client_df):
    """

    :param model: serialized model that has a predict_proba method
    :param client_df: (DataFrame)
    :return:
    :rtype: (tuple)
    """

    explainer = shap.TreeExplainer(
        model)  # TODO : LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
    shap_values = explainer.shap_values(client_df)
    return shap_values


def get_shap_expected_value(model, client_df):
    """

    """
    explainer = shap.TreeExplainer(
        model)  # TODO : LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray
    shap_values = explainer.shap_values(client_df)
    # ATTENTION : weird : if we call shap_values then expected value changes to a list of array / 0 and 1
    expected_values = explainer.expected_value
    return shap_values, expected_values  # array


def get_df_shap(shap_values, client_df):
    """

    :param shap_values:
    :param client_df: (DataFrame)
    :return:
    :rtype:
    """
    df_shap = pd.DataFrame({
        'SHAP value': shap_values[1][0],
        'feature': client_df.columns
    })
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)

    return df_shap
