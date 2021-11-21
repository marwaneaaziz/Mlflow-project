# Mlflow-project

In order to get in hand in Mlflow technology, to do so the project will be divided in three parts:

-Building Classical ML projects with respect to basic ML Coding best practices
-Integrate MLFlow to your project
-Integrate ML Interpretability to the project

The Dataset used to build the ML models is the Home Credit Risk Classification

Link: https://www.kaggle.com/c/home-credit-default-risk/data

This project is composed of 3 scripts : 


train.py :
Is the script where data is loaded, preprocessed, and three models are trained on the classification problem, Xgboost Classifier, Random Forest and Gradient Boosting, 
an mlflow experiment is started and a the three models are loged as well as their respective accuracy score and auc roc score.

Usinf the command: mlflow ui, we cann acces the user interface of mlflow and see the runs and the logs of the mlflow experiment.

The models are then served on a localhost server on a Rest API, using the following command in a command prompt : 
mlflow models serve -m xgboost -p 1234 -h 0.0.0.0
The user can choose which model to load xgboost,randomforest,gradientboosting


predict.py : 
This script loads the test dataset, and send a post request to the following adress : 'https//0.0.0.0.1234/invocations', which is the adress of the model api /invocations.


exp.ipynb : 
In this notebook using shap, we explain and interpret the developed ML models
