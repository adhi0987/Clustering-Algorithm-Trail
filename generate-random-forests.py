import os
import glob
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

def main():
    input_dir = "Processed_Users_ML"
    output_dir = "model"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}.")
        return
        
    METADATA_COLS = ['Age', 'Height', 'Weight', 'Gender', 'User_ID', 'Trial_ID', 'time']
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        username = filename.replace('_data.csv', '')
        print(f"Training Random Forest model for {username}...")
        
        df = pd.read_csv(file_path)
        
        # Locate target variables
        activity_cols = [c for c in df.columns if c.startswith('Activity_')]
        if not activity_cols:
            print(f" -> Skipping {username}, no activity columns found.")
            continue
            
        y = df[activity_cols].idxmax(axis=1)
        y = y.str.replace('Activity_', '')  
        
        # Build features excluding metadata and targets
        ignore_cols = METADATA_COLS + activity_cols
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        
        X = df[feature_cols]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 80-20 Train-Test split for model evaluation
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        # Test local validation testing score
        test_acc = rf.score(X_test_scaled, y_test)
        
        # Define output save configuration
        model_filename = os.path.join(output_dir, f"RF_model_{username}.pkl")
        scaler_filename = os.path.join(output_dir, f"scaler_{username}.pkl")
        
        # Save Scikit-Learn Model and standard scaler parameters mapping
        joblib.dump(rf, model_filename)
        joblib.dump(scaler, scaler_filename)
        
        print(f" -> Validation Accuracy: {test_acc * 100:.2f}% | Saved logic to: {model_filename}")
        
    print(f"\n[SUCCESS] Successfully trained, evaluated, and saved models/scalers explicitly targeting {len(csv_files)} datasets continuously into the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
