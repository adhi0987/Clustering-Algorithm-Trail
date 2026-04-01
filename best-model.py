import os
import glob
import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

class Logger(object):
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Suppress warnings from logistic regression/MLP not converging perfectly
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def compute_macro_fpr(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fprs = []
    # Loop over all classes
    for i in range(len(labels)):
        # False Positives for class i
        FP = cm[:, i].sum() - cm[i, i]
        # True Negatives for class i
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        
        if (FP + TN) == 0:
            fprs.append(0.0)
        else:
            fprs.append(FP / (FP + TN))
    return np.mean(fprs)

def evaluate_models():
    # Setup dual-logging to output.log and stdout
    sys.stdout = Logger("output.log")
    
    input_dir = "Processed_Users_ML"
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}. Please run the previous script first.")
        return
        
    print(f"Found {len(csv_files)} files to evaluate.")
    
    # Define our suite of models to evaluate, including Neural Network
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, n_jobs=-1, max_iter=200),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
    }
    
    # Store the results for printing at the end
    results = {name: {'wins': 0, 'file_metrics': []} for name in models.keys()}
    
    # Metadata columns that we specifically WANT TO EXCLUDE from feature training
    METADATA_COLS = ['Age', 'Height', 'Weight', 'Gender', 'User_ID', 'Trial_ID', 
                     'time', 'accel_time_list', 'gyro_time_list', 'orientation_time_list']
    
    # Records meant specifically to be populated to CSV later
    acc_records = []
    fpr_records = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\n--- Processing {filename} ---")
        df = pd.read_csv(file_path)
        
        # 1. Identify Target Label columns
        activity_cols = [c for c in df.columns if c.startswith('Activity_')]
        if not activity_cols:
            print(f"No Activity columns found in {filename}, skipping.")
            continue
            
        # Reverse one-hot encoding for the models
        y = df[activity_cols].idxmax(axis=1)
        
        # 2. Identify Feature columns
        ignore_cols = METADATA_COLS + activity_cols
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        
        X = df[feature_cols]
        
        # Clean numerical anomalies
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 3. Train/Test split: 80% used to train models, 20% to test them ON THE SAME CSV (same user)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Standard Scaler to help Neural Network and Logistic Regression converge faster
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        unique_labels = y.unique()
        
        best_model_name = None
        best_acc = -1
        
        row_acc = {'File': filename}
        row_fpr = {'File': filename}
        
        # 5. Train, predict, evaluate
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            fpr = compute_macro_fpr(y_test, y_pred, labels=unique_labels)
            
            # Save for CSV Export
            row_acc[name] = acc
            row_fpr[name] = fpr
            
            # Determine if it's the current best
            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                
            # Log results
            results[name]['file_metrics'].append({
                'file': filename,
                'acc': acc,
                'fpr': fpr
            })
            print(f"  {name:20s} -> Accuracy: {acc*100:6.2f}%,  Macro FPR: {fpr:6.4f}")
            
        print(f"  >> Best Model: {best_model_name} (Acc: {best_acc*100:.2f}%)")
        results[best_model_name]['wins'] += 1
        
        acc_records.append(row_acc)
        fpr_records.append(row_fpr)
        
    print("\n========================================================")
    print("                     FINAL SUMMARY")
    print("========================================================")
    
    overall_best_model = None
    max_wins = -1
    
    for name in models.keys():
        wins = results[name]['wins']
        print(f"\nModel [{name}] performed best on {wins} CSV file(s).")
        
        if wins > max_wins:
            max_wins = wins
            overall_best_model = name
            
        avg_acc = 0
        avg_fpr = 0
        metrics = results[name]['file_metrics']
        
        if metrics:
            print("  Each CSV File's Accuracy & False Positive Rate:")
            for m in metrics:
                print(f"    - {m['file']:15s}  |  Acc: {m['acc']*100:6.2f}%  |  FPR: {m['fpr']:6.4f}")
                avg_acc += m['acc']
                avg_fpr += m['fpr']
                
            avg_acc /= len(metrics)
            avg_fpr /= len(metrics)
            print(f"  -> AVERAGE for {name} |  Acc: {avg_acc*100:6.2f}%  |  FPR: {avg_fpr:6.4f}")

    print("\n========================================================")
    print("                     CONCLUSION")
    print("========================================================")
    print(f"Based on evaluating models on the movement dataset, the best overall model across all CSV computations is: '{overall_best_model}'.")
    
    # Export records to CSV format files
    df_acc_output = pd.DataFrame(acc_records)
    df_fpr_output = pd.DataFrame(fpr_records)
    
    # Save to OS mapping
    df_acc_output.to_csv('model_accuracies.csv', index=False)
    df_fpr_output.to_csv('model_fprs.csv', index=False)
    
    print("\n[SUCCESS] Saved comprehensive continuous Accuracy tables strictly structured to 'model_accuracies.csv'.")
    print("[SUCCESS] Saved comprehensive continuous FPR table strictly structured to 'model_fprs.csv'.")
    print("[SUCCESS] Complete terminal validation log and computations are successfully duplicated/appended chronologically to root log file -> 'output.log'.")

if __name__ == "__main__":
    evaluate_models()
