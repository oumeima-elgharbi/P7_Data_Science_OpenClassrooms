# P7_Data_Science_OpenClassrooms

#### C:\ProgramData\Anaconda3\python.exe -m pip install evidently

if you have packages problems with Streamlit :
#### C:\ProgramData\Anaconda3\python.exe -m pip install --upgrade --force-reinstall streamlit --user


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