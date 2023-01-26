# P7_Data_Science_OpenClassrooms : Back-end

#### Methods

- Using client id, we get a preprocessed client (meaning with all his data and history of loans)
  GET /clients/{client_id}/

- We compute the probability that the client might not repay the loan
  POST /predict/

- We compute Shapley values for each feature of client (local feature importance)
  POST /shap/

#### Run

If deploy == False :

- url : http://127.0.0.1:8000/
  Else:
- url : https://p7-data-science-openclassrooms.herokuapp.com/

To run the back-end from the root of the repository (we only have one repository for back and front)

**For dev purposes add --reload**

```bash
uvicorn back_end.main:app -- reload
```

We could have done like below if we wanted to separate the front-end from the back-end in two different repositories.

```bash
cd back_end
uvicorn main:app --reload 
```

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