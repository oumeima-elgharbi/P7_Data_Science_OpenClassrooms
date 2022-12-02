# Prediction API

### Context

### Postman

Using FastAPI : main.py

body : {"sid":710002,"pid":130690.0,"req_time":1538395839000,"o":"116.49,39.99","d":"116.39,39.91","click_time":
1538395843000,"click_mode":2.0,"plan_time":1538395839000,"
plans":"[{\"distance\": 18736, \"price\": 500, \"eta\": 4642, \"transport_mode\": 2}, {\"distance\": 17124, \"price\": \"\", \"eta\": 1936, \"transport_mode\": 3}, {\"distance\": 17124, \"price\": 4600, \"eta\": 2176, \"transport_mode\": 4}, {\"distance\": 17638, \"price\": 400, \"eta\": 5828, \"transport_mode\": 1}]"
}

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
