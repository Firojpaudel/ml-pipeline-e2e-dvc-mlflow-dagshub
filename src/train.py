import os 
import yaml 
from dotenv import load_dotenv
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from mlflow.models import infer_signature
from urllib.parse import urlparse

# loaing from the .env file
load_dotenv()

mlflow_tracking_uri= os.getenv('MLFLOW_TRACKING_URI')

os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
