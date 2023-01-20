#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from tqdm import tqdm

import joblib
from prediction_functions import read_yml  # FastAPI_app.
# from feature_engineering import generate_dataset

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")

global seed
seed = 42

print("__Getting config")
config = read_yml("config.yml")


def get_client(client_id, real_time=False):
    """

    :param client_id:
    :param real_time: if False, we return the preprocessed client's application
    :return:
    """
    print("__Getting client's application from database__")
    if not real_time:
        data = pd.read_csv(config["clients_database_preprocessed"])
        client = data[data["SK_ID_CURR"] == client_id]
    else:
        print("__Getting client's application from database__")
        data = pd.read_csv(config["clients_database"])
        client = data[data["SK_ID_CURR"] == client_id]
    return client


def preprocess_one_application(client_id, real_time=False):
    """

    :param client_id:
    :param real_time:
    :return:
    """
    if not real_time:
        # data = pd.read_csv(config["clients_database_preprocessed"])
        # preprocessed_client = data[data["SK_ID_CURR"] == client_id]
        preprocessed_client = get_client(client_id, real_time=False)
    else:
        print("__Getting client's application from database__")
        client = get_client(client_id, real_time=True)

        print("Preprocessing for selected client")

        print("Not working yet !")
        encoder_application = config["preprocessing"]["application"]
        encoder_bureau = config["preprocessing"]["bureau"]
        encoder_bureau_balance = config["preprocessing"]["bureau_balance"]
        encoder_credit_card = config["preprocessing"]["credit_card_balance"]
        encoder_pos = config["preprocessing"]["POS_CASH_balance"]
        encoder_previous_application = config["preprocessing"]["previous_application"]

        # preprocessed_client = generate_dataset(input_path="dataset/cleaned/",
        #                                       application_filename='one_query_test.csv',
        #                                       output_file="dataset/cleaned/preprocessed_one_query_test.csv",
        #                                       training=False)

    features = [f for f in preprocessed_client.columns if f not in ['SK_ID_CURR']]
    return preprocessed_client[features]
