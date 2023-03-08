# P7_Data_Science_OpenClassrooms : front-end

## I) Context

#### Dashboard specifications

Michaël vous a fourni des spécifications pour le dashboard interactif. Celui-ci devra contenir au minimum les
fonctionnalités suivantes :

Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une
personne non experte en data science.
Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de
clients similaires.

## II) Run

- url development :  **http://localhost:8501/**
- url production : **https://p7-data-science-oc-dashboard.herokuapp.com/**

To run the back-end from the root of the repository (we only have one repository for back and front)

We separated the front-end from the back-end in two different repositories.

```bash
cd frontend
streamlit run dashboard.py
```

#### Some infos

if you have packages problems with Streamlit :

````bash
$ C:\ProgramData\Anaconda3\python.exe -m pip install --upgrade --force-reinstall streamlit 
````

--user won't work in virtual environment

## III) Resources folder and config

We have a config file to find this resource :

- HomeCredit_columns_description.csv : this file contains descriptions for the columns in the various data files.

In the config, we also have :

- the threshold (limit to refuse the credit to a client).
- the endpoints from the back-end
