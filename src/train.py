import os 
import yaml 
from dotenv import load_dotenv
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# loading from the .env file
load_dotenv()
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

#! Load the params 
def load_params():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
    return params

#! Hyperparams tuning funciton 
def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')
    grid_search.fit(X_train, y_train)
    logging.info(f"Best params: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

#@ Now comes the training function
def train(data_path,model_path,ranadom_state,n_estimators,max_depth):
    params = load_params()
    
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).values.ravel()  # Flattened for sklearn
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).values.ravel()

    
