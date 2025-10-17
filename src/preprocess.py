import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Loading params from params.yaml 
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["preprocess"]

def preprocess(input_path, output_path):

    #! loading the raw data
    data = pd.read_csv(input_path)
    print(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")

    #! handling the missing values 
    if data.isnull().sum().sum() > 0:
        print("Missing values detected. Filling with the column mean ...")
        data.fillna(data.mean(numeric_only=True), inplace=True)
    
    #! dropping duplicates if any of them exist 
    data.drop_duplicates(inplace=True)

    #! features and target columns define
    target_col = params.get("target_col", "tsunami")
    X= data.drop(columns=[target_col])
    y= data[target_col]

    #! Then lets do train_test split
    test_size = params["test_size"]
    random_state = params["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    #! Feature Scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #! Then we convert back to dataframe for saving 
    X_train_df= pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df= pd.DataFrame(X_test_scaled, columns=X.columns)
    y_train_df= pd.DataFrame(y_train).reset_index(drop=True)
    y_test_df= pd.DataFrame(y_test).reset_index(drop=True)

    #! finally saving the processed files
    os.makedirs(output_path, exist_ok=True)
    X_train_df.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train_df.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test_df.to_csv(os.path.join(output_path, "y_test.csv"), index=False)

    print(f"Preprocessing complete. Files saved to {output_path}")

if __name__ == "__main__":
    preprocess(params["input"], params["output"])