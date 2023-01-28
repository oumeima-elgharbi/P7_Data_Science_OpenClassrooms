# P7_Data_Science_OpenClassrooms : back-end

Hello !

We used the framework FastAPI to develop an API for
the [dashboard](jetbrains://pycharm/navigate/reference?project=P7_Data_Science_OpenClassrooms&path=frontend/README.md).

## I) Methods

##### Doc : http://127.0.0.1:8000/docs

Using Postman or the dashboard ;)

- Using client id, we get a preprocessed client (meaning with all his data and history of loans)
  **GET /clients/{client_id}/**
  Body : {}
  => response : you will get a preprocessed client in json format.
  NB : request takes more than 10s, not optimized.

- Using client id in the body, we get a preprocessed client (meaning with all his data and history of loans)
  **POST /client_data/**
  Body : {"client_id": 1}

After getting the client's data, copy/paste the response as the body of the two following requests

- We compute the probability that the client might not repay the loan
  **POST /predict/**
  Body : the response from POST /client_data or GET / clients/{client_id}
  => response : you will get the proba that the client will repay the loan


- We compute Shapley values for each feature of client (local feature importance)
  **POST /shap/**
  Body : the response from POST /client_data or GET / clients/{client_id}
  => response : you will get the SHAP values for feature of your preprocessed client

## II) Run

- url development : **http://127.0.0.1:8000/**
- url production : **https://p7-data-science-oc-api.herokuapp.com/**

To run the back-end from the root of the repository (we only have one repository for back and front)

**For dev purposes add --reload**

We separated the front-end from the back-end in two different repositories.

```bash
cd back_end
uvicorn main:app --reload 
```

## III) Resources folder and config

We have a config file to find these two resources :

- data_test_preprocessed_vf.csv.gz
- model.joblib


