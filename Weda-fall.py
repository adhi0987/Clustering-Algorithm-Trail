import os
import re
import pandas as pd
import glob

def parse_readme(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    users = []
    activities = {}
    
    for line in lines:
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                # Participant tables
                if parts[1].isdigit() and len(parts) >= 6:
                    uid = int(parts[1])
                    if parts[2].isdigit() and parts[5].strip() in ['Male', 'Female']: # Age and Gender
                        users.append({
                            'User_id_str': f"U{uid:02d}",
                            'Age': float(parts[2]),
                            'Height': float(parts[3]),
                            'Weight': float(parts[4]),
                            'Gender': parts[5]
                        })
                
                # Activity tables
                if re.match(r'^(F|D)\d{2}$', parts[1]):
                    activities[parts[1]] = parts[2]
                    
    users_df = pd.DataFrame(users)
    return users_df, activities

def process_dataset():
    base_dir = r"WEDA-FALL\WEDA-FALL-main"
    readme_path = os.path.join(base_dir, "README.md")
    dataset_dir = os.path.join(base_dir, "dataset", "50Hz")
    
    if not os.path.exists(readme_path):
        print("README.md not found!")
        return
        
    users_df, activities = parse_readme(readme_path)
    
    print("=== Dataset Details Extracted from README ===")
    print(f"Total Participants found: {len(users_df)}")
    print(f"Total Activities found: {len(activities)}")
    print("\nSample Participants:")
    print(users_df.head())
    print("\nActivities mapping:")
    for k, v in list(activities.items())[:5]:
        print(f"  {k}: {v}")
    print("===========================================\n")
    
    # We want to output a CSV for each person
    # Keep track of all files associated with a person
    # file pattern: U01_R01_accel.csv in folder F01
    
    # dict: user_str -> list of dataframes
    user_dataframes = {}
    
    activity_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"Scanning dataset directories in {dataset_dir}...")
    
    # gather files
    # Dictionary mapping (user_str, activity, trial) -> {'accel': path, 'gyro': path, 'orientation': path}
    trials = {}
    
    for act in activity_dirs:
        act_path = os.path.join(dataset_dir, act)
        files = os.listdir(act_path)
        for f in files:
            if f.endswith('.csv'):
                # parse filename
                # format: U01_R01_accel.csv
                parts = f.replace('.csv','').split('_')
                if len(parts) == 3:
                    user_str, trial_str, sensor_type = parts
                    key = (user_str, act, trial_str)
                    if key not in trials:
                        trials[key] = {}
                    trials[key][sensor_type] = os.path.join(act_path, f)
                    
    print(f"Found {len(trials)} distinct trials to process.")
    
    OUTPUT_DIR = "Processed_Users"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process trials and group by user
    import sys
    
    processed_count = 0
    total_trials = len(trials)
    
    for (user_str, act, trial_str), sensors in trials.items():
        processed_count += 1
        # Need all three sensors generally, but we'll merge what we have
        df_accel = None
        df_gyro = None
        df_orient = None
        
        if 'accel' in sensors:
            df_accel = pd.read_csv(sensors['accel'])
            # Assuming time column is 'accel_time_list'
            time_col = [c for c in df_accel.columns if 'time' in c][0]
            df_accel = df_accel.rename(columns={time_col: 'time'})
            
        if 'gyro' in sensors:
            df_gyro = pd.read_csv(sensors['gyro'])
            time_col = [c for c in df_gyro.columns if 'time' in c][0]
            df_gyro = df_gyro.rename(columns={time_col: 'time'})
            
        if 'orientation' in sensors:
            df_orient = pd.read_csv(sensors['orientation'])
            time_col = [c for c in df_orient.columns if 'time' in c][0]
            df_orient = df_orient.rename(columns={time_col: 'time'})
            
        # Merge them based on nearest time
        merged_df = None
        
        dfs = [df_accel, df_gyro, df_orient]
        dfs = [df for df in dfs if df is not None]
        
        if not dfs:
            continue
            
        for d in dfs:
            d.sort_values('time', inplace=True)
            
        merged_df = dfs[0]
        for d in dfs[1:]:
            merged_df = pd.merge_asof(merged_df, d, on='time', direction='nearest', tolerance=0.1)
            
        merged_df['Activity_Code'] = act
        merged_df['Activity_Name'] = activities.get(act, "Unknown")
        merged_df['Trial_ID'] = trial_str
        
        if user_str not in user_dataframes:
            user_dataframes[user_str] = []
            
        user_dataframes[user_str].append(merged_df)
        
        # simple progress output
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{total_trials} trials...")

    print("Merging user data and saving...")
    
    for user_str, df_list in user_dataframes.items():
        user_final_df = pd.concat(df_list, ignore_index=True)
        
        # Merge physical metadata
        meta = users_df[users_df['User_id_str'] == user_str]
        if not meta.empty:
            for col in ['Age', 'Height', 'Weight', 'Gender']:
                user_final_df[col] = meta.iloc[0][col]
        else:
            for col in ['Age', 'Height', 'Weight', 'Gender']:
                user_final_df[col] = None
                
        user_final_df['User_ID'] = user_str
                
        out_csv = os.path.join(OUTPUT_DIR, f"{user_str}_data.csv")
        user_final_df.to_csv(out_csv, index=False)
        print(f"Saved {user_str} data to {out_csv} with {len(user_final_df)} rows.")

if __name__ == "__main__":
    process_dataset()
