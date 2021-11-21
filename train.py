

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
import os
import shutil


# Scoring method
def score(model,x_dev,y_dev) : 
    y_pred = model.predict(x_dev)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_dev, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    roc_auc = roc_auc_score(y_dev, predictions)
    print("ROC AUC: %.2f%%" % (roc_auc * 100.0)) 

    return accuracy,roc_auc

def prepare_data():

    app_train = pd.read_csv(r'application_train.csv')
    app_train = app_train.dropna()
    app_test = pd.read_csv(r'application_test.csv')
    app_test = app_test.dropna()

    # Calculation the number of NaN values and their %

    train = app_train.isnull().sum().sort_values(ascending=False).to_frame(name='number_of_missing_values')
    train['%'] = 100*train/len(app_train)
    train = train[train['number_of_missing_values']!=0]
    # Calculating the number of NaN values and their %

    test = app_test.isnull().sum().sort_values(ascending=False).to_frame(name='number_of_missing_values')
    test['%'] = 100*test/len(app_test)
    test = test[test['number_of_missing_values']!=0]

    # Checking the presence of duplicated data
    print("No Presence of duplicated data" if not (app_train.duplicated().any() and app_test.duplicated().any()) else print("Presence of duplicated data"))

    print('Training test shape: ', app_train.shape)
    print('Testing test shape: ', app_test.shape)

    train_features = train.loc[train['%'] >= 50]
    for col in train_features.index:
        app_train.drop([str(col)], axis = 1, inplace = True)
        app_test.drop([str(col)], axis = 1, inplace = True)

    print('Training test shape: ', app_train.shape)
    print('Testing test shape: ', app_test.shape)

    labelencoder = LabelEncoder()
    count = 0

    for col in app_train:
        if app_train[col].dtype == 'object':
            if len(list(app_train[col].unique())) <= 2:
                labelencoder.fit(app_train[col])
                app_train[col] = labelencoder.transform(app_train[col])
                app_test[col] = labelencoder.transform(app_test[col])
                
                count += 1
                
            
    print('%i column(s) were label encoded' %count)
    print('Training test shape: ', app_train.shape)
    print('Testing test shape: ', app_test.shape)

    app_train = pd.get_dummies(app_train)
    app_test = pd.get_dummies(app_test)

    print('Training test shape: ', app_train.shape)
    print('Testing test shape: ', app_test.shape)
    # We notice that the training set and the set don't have the same number of variables.
    # One hot encoding created more columns in the training set because there were some categorical variables with categories that
    # are not present in the test set
    # => We use the .align() method to align both sets and remove the columns that are not present in both sets using axis = 1 argument
    # We remove TARGET with align, but we keep track of it since it's only present  in the test set
#target = app_train['TARGET']

    #app_train, app_test = app_train.align(app_test, join='inner', axis=1)

    #app_train['TARGET'] = target

    # Since we have a dataset that is label/One_hot encoded, we can study the correlation between the variables
    # Mutual Info Classification

    target = app_train['TARGET']

    #variables = df.drop(['TARGET'], axis = 1)

    #df_features = pd.DataFrame({'Variables': [var for var in variables.columns.to_numpy()], 'Score': mutual_info_classif(variables, target)})

    #df_features = df_features.loc[df_features['Score'] >= 0.002]

    #app_train = variables[df_features['Variables'].to_numpy()] 
    #app_test = app_test[df_features['Variables'].to_numpy()].dropna()


    print('Training test shape: ', app_train.shape)
    print('Testing test shape: ', app_test.shape)
    # We Standardize the data
    


    
    app_train, app_test = app_train.align(app_test, join='inner', axis=1)
    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)
    scaler = StandardScaler()
    app_train = pd.DataFrame(scaler.fit_transform(app_train),columns = app_train.columns)

   
    
    x_test = pd.DataFrame(scaler.transform(app_test),columns = app_test.columns)
    x_train, x_dev, y_train, y_dev = train_test_split(app_train, target, test_size = 0.2, random_state = 42)
    print('Training Features shape: ', app_train.shape)
    print('Testing Features shape: ', app_test.shape)
    pd.DataFrame(x_test).to_csv("test_data.csv",index = False)
    pd.DataFrame(x_test).to_json("test_data.json",orient = "columns")

    return( x_train, x_dev, y_train, y_dev,x_test)

if __name__ == "__main__" : 
    warnings.filterwarnings("ignore")
    mlflow.set_experiment(experiment_name="mlflow exp")

    x_train, x_dev, y_train, y_dev,x_test = prepare_data()
        # Models
    # 1. XGBOOST
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)
    accuracy,roc_auc  = score(xgb_model,x_dev,y_dev)
    mlflow.log_metric("xgb_accuracy",accuracy)
    mlflow.log_metric("xgb_roc_auc",roc_auc)


    print("Accuracy: %.2f%%" % (accuracy * 100.0),("ROC AUC: %.2f%%" % (roc_auc * 100.0)))

    # 2. Random Forest Classifier
    rfc_model = RandomForestClassifier()
    rfc_model.fit(x_train,y_train)
    accuracy,roc_auc  = score(rfc_model,x_dev,y_dev)
    mlflow.log_metric("rfc_accuracy",accuracy)
    mlflow.log_metric("rfc_roc_auc",roc_auc)

    print("Accuracy: %.2f%%" % (accuracy * 100.0),("ROC AUC: %.2f%%" % (roc_auc * 100.0)))


    # 3. GradientBoosting Classifier
    gb_model = GradientBoostingClassifier()
    gb_model.fit(x_train,y_train)
    accuracy,roc_auc = score(gb_model,x_dev,y_dev)
    mlflow.log_metric("gb_accuracy",accuracy)
    mlflow.log_metric("gb_roc_auc",roc_auc)

    print("Accuracy: %.2f%%" % (accuracy * 100.0),("ROC AUC: %.2f%%" % (roc_auc * 100.0)))

    model_types = ["xgboost","randomforest","gradientboosting"]
    for i in model_types : 
        if os.path.exists(i):
            shutil.rmtree(i)
    
    xgb_model_path = "xgb_model.pth"
    xgb_model.save_model(xgb_model_path)
    mlflow.pyfunc.save_model(xgb_model,"xgboost",artifacts = {
    "xgb_model": xgb_model_path
})
    mlflow.sklearn.save_model(rfc_model,"randomforest")
    mlflow.sklearn.save_model(gb_model,"gradientboosting")
