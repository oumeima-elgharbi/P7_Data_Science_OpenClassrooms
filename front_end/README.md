# P7_Data_Science_OpenClassrooms : Front-end

#### Run

If deploy == False :

- url : http://localhost:8501/
  Else:
- url : https://p7-data-science-openclassrooms.herokuapp.com/

To run the front-end from the root of the repository (we only have one repository for back and front)

```bash
streamlit run front_end/dashboard.py
```

We could have done like below if we wanted to separate the front-end from the back-end in two different repositories.

```bash
cd front_end
streamlit run dashboard.py
```

#### Heroku-22 stack

#### Files : Procfile and setup.sh

- Streamlit :
  web: sh setup.sh && streamlit run dashboard.py

We need a setup.sh to run streamlit dashboard

To open the web service at :

### Config front-end

- threshold
- **HomeCredit_columns_description.csv** : this file contains descriptions for the columns in the various data files.

######     

if you have packages problems with Streamlit :

#### C:\ProgramData\Anaconda3\python.exe -m pip install --upgrade --force-reinstall streamlit --user

#### Spécifications du dashboard

Michaël vous a fourni des spécifications pour le dashboard interactif. Celui-ci devra contenir au minimum les
fonctionnalités suivantes :

Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une
personne non experte en data science.
Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de
clients similaires.

Prediction
St
SHAP
bouger chiffre client et changer var globale (exemple : income)

