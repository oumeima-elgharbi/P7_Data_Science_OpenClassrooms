# P7_Data_Science_OpenClassrooms

Hello !

Welcome to this repository.

This README contains information about :

- I) Context
- II) Virtual environment
- III) Deployment (Cloud : Heroku)

# ---------------------------------------------------------------------

## I) Context

Home credit default : Kaggle

Supervised machine learning problem.

Data Science
=> [model_development ](jetbrains://pycharm/navigate/reference?project=P7_Data_Science_OpenClassrooms&path=model_development/README.md)

- Exploratory Data Analysis
- SMOTE, GridsearchCV, Cross-Validation
- Feature importance : global and local (Shapley value)
- Method pdf

MLOps :

- Dashboard (Streamlit)
  => [front-end](jetbrains://pycharm/navigate/reference?project=P7_Data_Science_OpenClassrooms&path=frontend/README.md)
- API (FastAPI)
  => [back-end](jetbrains://pycharm/navigate/reference?project=P7_Data_Science_OpenClassrooms&path=backend/README.md)
- Cloud deployment (Heroku)

The API can be found at this url : https://p7-data-science-oc-api.herokuapp.com/
The dashboard can be found here : https://p7-data-science-oc-dashboard.herokuapp.com/

## II) Virtual environment

##### Requirements

- Run pipreqs for front-end, back-end, model preparation to get a requirements.txt

````bash
pipreqs
````

##### 1) Install python

Download python 3.10.9 and install it : https://www.python.org/downloads/release/python-3109/

##### 2) Installation steps !!!!!!

- to get a virtual env run the following
- to activate it (you get (venv) in the terminal : venv\Scripts\activate.bat ((or venv\Scripts\activate))
- to install the requirements with the right versions
- to install Jupyter

````bash
virtualenv --python C:\Users\oumei\AppData\Local\Programs\Python\Python310\python.exe venv
venv\Scripts\activate.bat 
pip install -r requirements.txt
pip install jupyter notebook
````

ATTENTION python 3.11 not supported by shap
we need Python 3.11.1 for Heroku-22 but not compatible with shap
Thus python 3.10.9

#### 3) Verification

Check that you have (venv) in your terminal

- to get the list of packages installed in **venv**
- to get the python version
- to get the python version that is run locally

````bash
pip list
python --version
py -V
python -V
````

#### 4) Add runtime.txt

runtime.txt contains the python version, run like below to check the version

````bash
cat runtime.txt
python -V
````

python -V to check which version of Python is being run locally

#### 5) Some infos

if you have packages problems with Streamlit :

````bash
$ C:\ProgramData\Anaconda3\python.exe -m pip install --upgrade --force-reinstall streamlit 
````

--user won't work in virtual environment

## III) Deployment

### 1.1) How to deploy the two heroku apps

We made the choice to have one Git repository for two web applications.

- back-end : the API
- front-end : the dashboard

For deployment we chose Heroku.
With Heroku, one heroku app means one web application.

We need two applications, the process to deploy the front-end and back-end apps is explained below.

#### First time running repo

- connect to heroku using Heroku CLI
- go to git repository (locally)

- create heroku app with the name wanted for the two apps

- For the first app :
    - create a remote branch with wanted name
    - push changes (after making any commit / commit that was also push to your origin master)

NB : with this method we are deploying a particular directory (backend or frontend in our case)
Thus, the app must be run as if we are in the directory backend (respectively frontend).

- Do the same for the second app

````bash
heroku login
cd Documents\OC_projets\P7\P7_Data_Science_OpenClassrooms

$ heroku create -a p7-data-science-oc-api --remote heroku-back
$ heroku create -a p7-data-science-oc-dashboard --remote heroku-front

$ heroku git:remote --remote heroku-back -a p7-data-science-oc-api
$ heroku git:remote --remote heroku-front -a p7-data-science-oc-dashboard

$ git subtree push --prefix backend heroku-back master
$ git subtree push --prefix frontend heroku-front master
````

#### To deploy your changes

Make some changes to the code you just cloned and deploy them to Heroku using Git.

NB : do we need to do git:remote before pushing ?

````bash
$ heroku git:remote --remote heroku-back -a p7-data-science-oc-api
$ heroku git:remote --remote heroku-front -a p7-data-science-oc-dashboard
````

````bash
$ git subtree push --prefix backend heroku-back master
$ git subtree push --prefix frontend heroku-front master
````

#### Helpful commands

- to delete remote branches :
    - list all remote branches
    - delete remote branch

````bash
$ git remote -v
$ git remote rm branch_name
````

If we had one repo and one app :

````bash
heroku login
cd Documents\OC_projets\P7\P7_Data_Science_OpenClassrooms

heroku git:remote -a my_heroku_app_name
git push my_heroku_remote_branch_name master
````

### 1.2) Google Cloud Storage

#### On Google Cloud

- project_name : "project-7-oc-376509"
- bucket_name : "p7-data"
- save files in a folder (blob) "resources"

- grant access to the bucket to the service account created

#### On Heroku : config vars

Set up two config variables for each app

- GOOGLE_APPLICATION_CREDENTIALS : google-credentials.json
- GOOGLE_CREDENTIALS : {your_json_service_account}

The service account cannot be committed for security reasons !

#### With the terminal

First, add this heroku google-buildpack for the two apps (the one below works for Heroku-22)
This enables the creation a service account saved in a json file :

````bash
heroku login
cd Documents\OC_projets\P7\P7_Data_Science_OpenClassrooms

heroku buildpacks:add https://github.com/gerywahyunugraha/heroku-google-application-credentials-buildpack --app p7-data-science-oc-api
heroku buildpacks:add https://github.com/gerywahyunugraha/heroku-google-application-credentials-buildpack --app p7-data-science-oc-dashboard
````

#### Locally

Save the json service account named google-credentials.json to test your apps locally.

Or, connect to your Google account using gcloud :

````bash
gcloud auth login
gcloud auth application-default login
````

Then, run the script to download data !

````bash
cd backend
python script_download_data_folder.py
````

````bash
cd frontend
python script_download_data_folder.py
````

- $gcloud config set project project-7-oc-376509
- $gcloud auth revoke --all # to log out of gcloud SDK

### 2) Information on Heroku-22 stack

#### Files : Procfile and setup.sh

- API :
  web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}

- Streamlit :
  web: sh setup.sh && streamlit run dashboard.py

We need a setup.sh to run streamlit dashboard. We use the url from the back-end in the dashboard.

# -----------------------------------------------------------------------------

TODO clean code :

- download resources when building the app (not when running the webservices)
- common folder for back and front : factorize !!
- unzip resources folder like before ?

- preprocess in real time for test and train ?? (train to think about solution)

- remove runtime.txt in root only ?
- slugignore in root ony ?

Later :

- pytest / unittest
- Preprocess one client
- MLFlow track experiments
- MLFlow registry
- Evidently