# P7_Data_Science_OpenClassrooms

Add readme for frontend and for backend

#### Virtual environment

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

#### Heroku-22 stack

#### Files : Procfile and setup.sh

- API :
  web: uvicorn back_end.main:app --host=0.0.0.0 --port=${PORT:-5000}

- Streamlit :
  web: sh setup.sh && streamlit run dashboard.py

We need a setup.sh to run streamlit dashboard

Using Windows CMD, go to repo with cd..

````bash
heroku login
cd Documents\OC_projets\P7\P7_Data_Science_OpenClassrooms
heroku git:remote -a p7-data-science-openclassrooms
git push heroku master
````

````bash
web: sh -c 'cd ./front_end/ && sh setup.sh && cd .. && streamlit run front_end/dashboard.py'
server: uvicorn back_end.main:app --host=0.0.0.0 --port=${PORT:-5000}

web: sh setup.sh && streamlit run front_end/dashboard.py
````


Clone the repository
Use Git to clone p7-data-science-openclassrooms's source code to your local machine.

````bash
$ heroku git:clone -a p7-data-science-openclassrooms
$ cd p7-data-science-openclassrooms
````

Deploy your changes
Make some changes to the code you just cloned and deploy them to Heroku using Git.

````bash
$ git add .
$ git commit -am "deploy True again 3!"
$ git push heroku master
````

To open the web service at : https://p7-data-science-openclassrooms.herokuapp.com/

````bash
heroku open
````

######     

#### C:\ProgramData\Anaconda3\python.exe -m pip install evidently

if you have packages problems with Streamlit :

#### C:\ProgramData\Anaconda3\python.exe -m pip install --upgrade --force-reinstall streamlit --user

##### Reference : https://leandeep.com/datalab-kaggle/kb002.html

For Wednesday

- Dashboard (Streamlit)
- SHAP : Feature importance

Thursday :

- Cloud (Heroku, Azure Web App)
- unittest / pytest

Friday :

- SMOTE
- CV + GridSearch
- EDA + model : Kaggle

Saturday :

- notes : precision / recall

Later :

- Preprocess one client
- MLFlow track experiments
- MLFlow registry
- Evidently

### Dataset

source : https://www.kaggle.com/c/home-credit-default-risk/data
download : https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip

#### Dataset Description

- **application_{train|test}.csv**

    - This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
    - Static data for all applications. One row represents one loan in our data sample.


- **bureau.csv**

    - All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for
      clients
      who have a loan in our sample).

    - For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before
      the
      application date.


- **bureau_balance.csv**

    - Monthly balances of previous credits in Credit Bureau.

    - This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the
      table
      has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the
      previous credits) rows.


- **POS_CASH_balance.csv**

    - Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
    - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash
      loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * #
      of months in which we have some history observable for the previous credits) rows.

- **credit_card_balance.csv**

    - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
    - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash
      loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards
        * # of months where we have some history observable for the previous credit card) rows.

- **previous_application.csv**

    - All previous applications for Home Credit loans of clients who have loans in our sample.
    - There is one row for each previous application related to loans in our data sample.


- **installments_payments.csv**
    - Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
    - There is
        - a) one row for every payment that was made plus
        - b) one row each for missed payment.
    - One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one
      previous Home Credit credit related to loans in our sample.

- **HomeCredit_columns_description.csv**

    - This file contains descriptions for the columns in the various data files.