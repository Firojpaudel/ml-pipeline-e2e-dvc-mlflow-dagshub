import os
import yaml
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["train"]

def evaluate():
    params = load_params()
    
    # Load test data and model
    data_path = params['data']
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()
    
    with open(params['model'], 'rb') as f:
        model = pickle.load(f)
    
    logging.info(f"Evaluating model on test set: {X_test.shape}")
    
    # Start MLflow evaluation run
    with mlflow.start_run(run_name="model_evaluation"):
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        # Add ROC-AUC if binary classification
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Log all metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            logging.info(f"{metric_name}: {value:.3f}")
        
        # Generate and log confusion matrix
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
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        
        # Feature importance plot (top 10)
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
        
        # Model quality assessment
        if metrics['f1'] > 0.7:
            mlflow.set_tag("quality_status", "production_ready")
            logging.info("✓ Model meets production quality threshold")
        else:
            mlflow.set_tag("quality_status", "needs_improvement")
            logging.info("⚠ Model F1 below threshold - consider retraining")
        
        logging.info("Evaluation complete. Artifacts logged to MLflow.")
        return metrics

if __name__ == "__main__":
    metrics = evaluate()
    print(f"Final F1 Score: {metrics['f1']:.3f}")