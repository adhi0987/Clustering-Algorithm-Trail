import os
import pandas as pd
import glob

def onehot_encode_directory():
    input_dir = "Processed_Users"
    output_dir = "Processed_Users_ML" # Saving to a new folder to preserve the original CSVs
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files. Starting one-hot encoding...")
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        # 1. One-hot encode Activity_Code
        if 'Activity_Code' in df.columns:
            df = pd.get_dummies(df, columns=['Activity_Code'], prefix='Activity')
            
        # 2. Map Gender to fully numerical values (0: Male, 1: Female)
        if 'Gender' in df.columns:
            df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
            
        # 3. Drop Activity_Name (since it's a long non-numerical string string and we already have it one-hot encoded)
        if 'Activity_Name' in df.columns:
            df.drop(columns=['Activity_Name'], inplace=True)
            
        # 4. Make Trial_ID numerical (e.g., "R01" -> 1)
        if 'Trial_ID' in df.columns:
            df['Trial_ID'] = df['Trial_ID'].astype(str).str.replace('R', '', regex=False)
            df['Trial_ID'] = pd.to_numeric(df['Trial_ID'], errors='coerce')
            
        # 5. Make User_ID numerical (e.g., "U01" -> 1)
        if 'User_ID' in df.columns:
            df['User_ID'] = df['User_ID'].astype(str).str.replace('U', '', regex=False)
            df['User_ID'] = pd.to_numeric(df['User_ID'], errors='coerce')

        # Convert any newly created boolean columns from get_dummies to integers (0 and 1)
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
                
        out_path = os.path.join(output_dir, os.path.basename(file_path))
        df.to_csv(out_path, index=False)
        print(f"Processed: {os.path.basename(file_path)}")
        
    print("\nEncoding Complete!")
    print(f"All data is now strictly numerical and saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    onehot_encode_directory()
