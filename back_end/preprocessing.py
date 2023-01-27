#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc

from utils import *  # FastAPI_app.
# from feature_engineering import generate_dataset

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")

global seed
seed = 42

print("__Preprocessing : getting config")
config_back = read_yml("back_end/config_backend.yml")
##from main import config_back ## TODO remove / trying to save memory...


def get_client_from_database(client_id, real_time=False):
    """

    :param client_id:
    :param real_time: if False, we return the preprocessed client's application
    :return:
    """
    print("__Getting client's application from database__")
    if not real_time:
        print("HERE1")
        gc.collect()
        data = pd.read_csv(config_back["clients_database_preprocessed"]) # MEMORY PB / TODO refacto and clean code !!
        print("HERE2")
        client = data[data["SK_ID_CURR"] == client_id]
        print("HERE3")
    else:
        print("__Getting client's application from database__")
        data = pd.read_csv(config_back["clients_database"])
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
        #preprocessed_client = get_client_from_database(data, client_id, real_time=False)
        preprocessed_client = get_client_from_database(client_id, real_time=False)
    else:
        print("__Getting client's application from database__")
        client = get_client_from_database(client_id, real_time=True)

        print("Preprocessing for selected client")

        print("Not working yet !")
        encoder_application = config_back["preprocessing"]["application"]
        encoder_bureau = config_back["preprocessing"]["bureau"]
        encoder_bureau_balance = config_back["preprocessing"]["bureau_balance"]
        encoder_credit_card = config_back["preprocessing"]["credit_card_balance"]
        encoder_pos = config_back["preprocessing"]["POS_CASH_balance"]
        encoder_previous_application = config_back["preprocessing"]["previous_application"]

        # preprocessed_client = generate_dataset(input_path="dataset/cleaned/",
        #                                       application_filename='one_query_test.csv',
        #                                       output_file="dataset/cleaned/preprocessed_one_query_test.csv",
        #                                       training=False)
    # to remove the column with the id
    return preprocessed_client.iloc[:, 1:]
