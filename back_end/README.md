# Prediction API

### I) Context

### II) Before starting

#### Requirements

```bash
pip install -r requirements.txt
```

#### Dataset folder

#### Data folder

- dataset/cleaned/data_test_preprocessed_vf.csv

#### Models

##### Preprocessing

- models

##### Prediction

- LGBMClassifier

### III) Run

#### 1) Web Service / POSTMAN

```bash
cd back_end
uvicorn main:app --reload
```

Doc : http://127.0.0.1:8000/docs

Using POSTMAN

- GET http://localhost:8000/clients/100001
  => response : you will get a preprocessed client in json format.

After getting the client's data, copy/paste the response as the body of the two following requests

- POST http://localhost:8000/predict/
  => response : you will get the proba that the client will repay the loan

- POST http://localhost:8000/shap/
  => response : you will get the SHAP values for feature of your preprocessed client

#### 2) Dashboard

url : http://localhost:8501/

```bash
cd front_end
streamlit run dashboard.py
```