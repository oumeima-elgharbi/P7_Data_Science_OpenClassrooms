import time
import pandas as pd
import matplotlib.pyplot as plt

nb_chunks = [i for i in range(500, 300000, 10000)]
nb_chunks

DATA_ALL_CLIENTS_PATH = r"C:\Users\oumei\Documents\OC_projets\P7\P7_Data_Science_OpenClassrooms\frontend\resources\data_train_preprocessed_vf.csv.gz"
times = []

columns_list = ['PAYMENT_RATE',
                'EXT_SOURCE_1',
                'EXT_SOURCE_3',
                'EXT_SOURCE_2',
                'DAYS_BIRTH',
                'DAYS_EMPLOYED',
                'AMT_ANNUITY',
                'DAYS_ID_PUBLISH',
                'APPROVED_CNT_PAYMENT_MEAN',
                'ACTIVE_DAYS_CREDIT_MAX',
                'SK_ID_CURR',
                'TARGET']

for n in nb_chunks:
    DATA_ALL_CLIENTS_CHUNKS = []

    t0 = time.time()
    with pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                     usecols=columns_list,
                     chunksize=n) as reader:
        for i, data in enumerate(reader):
            # print(i, "_Update the list of chunks_ Shape : ", data.shape)
            DATA_ALL_CLIENTS_CHUNKS.append(data)
    t1 = time.time() - t0
    print(t1, "seconds")
    times.append(t1)
print("END")



t0 = time.time()
data = pd.read_csv(DATA_ALL_CLIENTS_PATH, encoding="utf-8", index_col="SK_ID_CURR",
                   usecols=columns_list)
t1 = time.time() - t0
print("HERE", t1, "seconds")

plt.plot(nb_chunks, times)
plt.show()