import requests


def request_client_data(model_uri, client_id):
    """

    :param model_uri:
    :param client_id: (int)
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json={"client_id": client_id})

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def request_prediction(model_uri, client_json):
    """

    :param model_uri:
    :param client_json:
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=client_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def request_shap(model_uri, client_json):
    """

    :param model_uri:
    :param client_json:
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=client_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


##############################################################"""

def request_shap_expected(model_uri, client_json):  # TODO refacto / like above just different response
    """

    :param model_uri:
    :param client_json:
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=client_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


##############################################################"""

def request_feature_importance(model_uri, client_json):  # TODO refacto / like above just different response
    """

    :param model_uri:
    :return:
    """
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=client_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()
