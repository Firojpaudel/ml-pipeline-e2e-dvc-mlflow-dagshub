import os 
import yaml 
from dotenv import load_dotenv
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle 
from sklearn.metrics import accuracy_score, f1_score
import joblib
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
def train(params):
    params = load_params()

    data_path = params['data_path']
    model_path = params['model_path']
    random_state = params['random_state']
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).values.ravel()  # Flattened for sklearn
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).values.ravel()

    with mlflow.start_run():
        mlflow.sklearn.autolog() #! This autologs params, metrics and model 

        if params.get('tune_hyperparams', False):
            param_grid = params.get('param_grid', {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, None]
            })
            model, best_params = hyperparameter_tuning(X_train, y_train, param_grid)
            mlflow.log_params(best_params)
        else:
            # Use parameters from YAML
            rf_params = {
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'random_state': params['random_state'],
                'class_weight': params.get('class_weight', 'balanced'),
                'n_jobs': params.get('n_jobs', -1),
                'oob_score': params.get('oob_score', True),
                'min_samples_split': params.get('min_samples_split', 2),
                'min_samples_leaf': params.get('min_samples_leaf', 1)
            }
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y_train)
        
        #! Then we evaluate on test set
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        accuracy= accuracy_score(y_test, y_pred)

        ## logging the metrics 
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_accuracy", accuracy)

        # Log parameters to MLflow
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("model_path", params['model_path'])
        for key, value in rf_params.items():
            mlflow.log_param(f"rf_{key}", value)
        
        #! Then we create the model signature
        predictions = model.predict(X_test)
        signature = infer_signature(X_test, predictions)

        # Save model locally using joblib
        os.makedirs(os.path.dirname(params['model_path']), exist_ok=True)
        joblib.dump(model, params['model_path'])

        # Log model file as artifact instead of using log_model()
        mlflow.log_artifact(params['model_path'], artifact_path="model")
        
        logging.info(f"Model saved to {params['model_path']}")
        logging.info(f"Test F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
        
        return model

if __name__ == "__main__":
    params = load_params()
    train(params)