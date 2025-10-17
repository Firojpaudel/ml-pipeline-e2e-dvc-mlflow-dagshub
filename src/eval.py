import os
import yaml
from dotenv import load_dotenv
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib  # Alternative to pickle

# Load MLflow config
load_dotenv()
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params():
    """Load train parameters (eval uses same paths)"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    train_params = params["train"]
    return train_params

def load_model_safely(model_path):
    """Safely load model with multiple fallback methods"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"Model file is empty: {model_path}")
    
    try:
        # Try pickle first
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully with pickle")
        return model
    except Exception as e:
        logging.warning(f"Pickle failed: {e}. Trying joblib...")
        try:
            # Try joblib as fallback
            model = joblib.load(model_path)
            logging.info("Model loaded successfully with joblib")
            return model
        except Exception as e2:
            raise RuntimeError(f"Failed to load model with pickle or joblib: {e2}")

def evaluate():
    params = load_params()
    
    # Use correct parameter names
    data_path = params['data_path']
    model_path = params['model_path']
    
    # Verify data files exist
    test_files = ['X_test.csv', 'y_test.csv']
    for file in test_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test data missing: {file_path}")
    
    # Load test data
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()
    
    # Load model with error handling
    try:
        model = load_model_safely(model_path)
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        logging.info("Please run 'python src/train.py' first to train the model")
        return None
    
    logging.info(f"Evaluating model on test set: {X_test.shape}")
    
    # Start MLflow evaluation run
    with mlflow.start_run(run_name="model_evaluation"):
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except:
                pass
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        # Add ROC-AUC if possible
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            logging.info(f"{metric_name}: {value:.3f}")
        
        # Log parameters
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("model_path", model_path)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Clean up
        if os.path.exists(cm_path):
            os.remove(cm_path)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        os.remove(report_path)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:][::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Top 10 Feature Importances")
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), 
                      [X_test.columns[i] for i in indices], 
                      rotation=45, ha='right')
            plt.tight_layout()
            importance_path = "feature_importance.png"
            plt.savefig(importance_path, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact(importance_path)
            os.remove(importance_path)
        
        # Quality assessment
        with open("params.yaml", "r") as f:
            all_params = yaml.safe_load(f)
        eval_thresholds = all_params.get("evaluate", {}).get("metrics_thresholds", {})
        f1_threshold = eval_thresholds.get('f1_min', 0.7)
        
        if metrics['f1'] > f1_threshold:
            mlflow.set_tag("quality_status", "production_ready")
            logging.info(f"✓ Model meets F1 threshold: {metrics['f1']:.3f} > {f1_threshold}")
        else:
            mlflow.set_tag("quality_status", "needs_improvement")
            logging.info(f"⚠ Model F1 {metrics['f1']:.3f} below threshold {f1_threshold}")
        
        logging.info("Evaluation complete.")
        return metrics

if __name__ == "__main__":
    metrics = evaluate()
    if metrics:
        print(f"Final F1 Score: {metrics['f1']:.3f}")
    else:
        print("Evaluation failed - please train model first")