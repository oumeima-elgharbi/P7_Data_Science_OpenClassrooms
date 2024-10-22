import gc

from utils import *  # FastAPI_app.
# from feature_engineering import generate_dataset

import warnings

# warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="once")

print("_____Preprocessing : getting config_____")
config_back = read_yml("config_backend.yml")


def get_client_from_database(client_id, database_name):  # data
    """
    if real_time = True, database_name = "clients_database" for new clients
    :param client_id:
    :param database_name:
    :return:
    """
    print("__Getting client's application from database__")
    with pd.read_csv(config_back[database_name], index_col="SK_ID_CURR", chunksize=10000) as reader:
        for data in reader:
            # print("\nHERE :", data.info(verbose=False, memory_usage="deep"), end="\n\n")
            if client_id in data.index:
                client = data[data.index == client_id]
                return client
            gc.collect()
    # if we get here it means the client was not found in the database
    print("__Client not found in database__")
    gc.collect()  # collects for last loop
    raise Exception("Client not in database")


def preprocess_one_application(client_id, database_name="new_clients_database_preprocessed", real_time=False):  # data
    """

    :param client_id:
    :param database_name:
    :param real_time:
    :return:
    """
    if not real_time:  # TODO add option for database name
        preprocessed_client = get_client_from_database(client_id, database_name)
    else:
        print("__Getting client's application from database__")
        client = get_client_from_database(client_id, database_name)

        preprocessed_client = {}

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
    return preprocessed_client  # .iloc[:, 1:] # we set the id as index
