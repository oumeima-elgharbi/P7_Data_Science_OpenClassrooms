# Prediction API

### Context

#### Dashboard 
url : http://localhost:8501/

##### run :
```bash
streamlit run dashboard.py

```

#### Web Service / POSTMAN

```bash
cd FastAPI_app
uvicorn main:app --reload
```

Doc : http://127.0.0.1:8000/docs

POST http://localhost:8000/v1/predict/client/100001


### Postman

Using FastAPI : main.py

body : {}

```bash
cd app
uvicorn main:app --reload
```

POST http://localhost:8000/predict

(for Flask : POST http://localhost:8080/predict)

### Before starting

#### Requirements

```bash
pip install -r requirements.txt
```

??? check syntax

### Data

#### Dataset folder

#### Data folder

- ##### preprocessing
    - input
    - output

- ##### prediction
    - output

### Models

##### Preprocessing

- models

##### Prediction

- LGBMClassifier

### Execution

To launch a local server ?

```bash
cd app
uvicorn main:app --reload
```
