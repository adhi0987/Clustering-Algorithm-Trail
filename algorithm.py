import os
import glob
import joblib
import numpy as np

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(vec1, vec2):
    # cosine_similarity expects 2D arrays, so we wrap the 1D arrays in lists
    return cosine_similarity([vec1], [vec2])[0][0]

def print_cluster_representatives(cluster_reps):
    print("\nCurrent Cluster Representatives:")
    for cid, rep in cluster_reps.items():
        print(f"  Cluster {cid}: {np.array2string(rep, precision=4, separator=', ')}")
    print()

def main():
    model_dir = "model"
    # Threshold for similarity to join an existing cluster.
    # The user can change this threshold between 0.9 and 1.0.
    THRESHOLD = 1.0
    
    # Get all model files
    model_files = sorted(glob.glob(os.path.join(model_dir, "RF_model_*.pkl")))
    
    if not model_files:
        print(f"No models found in {model_dir}/ directory.")
        return
        
    print(f"Found {len(model_files)} models. Starting clustering algorithm with threshold {THRESHOLD}...\n")
    
    
    clustering_dir = "clustering_results_"+str(THRESHOLD)
    os.makedirs(clustering_dir, exist_ok=True)
    
    clusters = {} # cluster_id -> list of usernames
    cluster_reps = {} # cluster_id -> average feature importance vector
    model_importances = {} # username -> feature importance vector
    
    cluster_counter = 1
    
    for file_path in model_files:
        filename = os.path.basename(file_path)
        # Extract the username which is just after 'RF_model_'
        username = filename.replace('RF_model_', '').replace('.pkl', '')
        
        print(f"Processing model for User: {username}")
        
        # Load model explicitly with joblib
        model = joblib.load(file_path)
        
        # Ensure we can extract feature importances
        if not hasattr(model, 'feature_importances_'):
            print(f" -> Skipping {username}, model doesn't have feature_importances_.")
            continue
            
        importance = model.feature_importances_
        model_importances[username] = importance
        
        # Find best matching existing cluster representative
        best_cluster = None
        best_sim = -1
        
        for cid, rep in cluster_reps.items():
            sim = calculate_similarity(importance, rep)
            if sim >= THRESHOLD and sim > best_sim:
                best_sim = sim
                best_cluster = cid
                
        if best_cluster is not None:
            # Add to the most similar cluster
            clusters[best_cluster].append(username)
            print(f" -> Matches Cluster {best_cluster} (Similarity: {best_sim:.4f}). Added to Cluster {best_cluster}.")
            
            # Update the cluster representative (average of all models in the cluster)
            all_importances_in_cluster = [model_importances[u] for u in clusters[best_cluster]]
            cluster_reps[best_cluster] = np.mean(all_importances_in_cluster, axis=0)
        else:
            # Create a new cluster and assign this model as the cluster representative
            clusters[cluster_counter] = [username]
            cluster_reps[cluster_counter] = importance
            
            # Print similarity if there were other clusters to compare
            sim_str = f"Max similarity was {best_sim:.4f}" if len(cluster_reps) > 1 else "First model in server"
            print(f" -> No similar clusters found ({sim_str}). Created new Cluster {cluster_counter}.")
            
            cluster_counter += 1
            
        print_cluster_representatives(cluster_reps)
            
        # Export current state after adding this model
        current_data = {}
        for cid, users in clusters.items():
            current_data[f"Cluster {cid}"] = users
            
        df_iter = pd.DataFrame.from_dict(current_data, orient='index').T
        df_iter = df_iter.fillna("") 
        
        iter_csv = os.path.join(clustering_dir, f"cluster_results_after_{username}.csv")
        df_iter.to_csv(iter_csv, index=False)
            
    # Prepare and Export Output
    print("\n[SUCCESS] Clustering complete. Generating CSV...")
    
    output_data = {}
    for cid, users in clusters.items():
        output_data[f"Cluster {cid}"] = users
        
    # Convert clustered users to DataFrame for CSV processing
    df = pd.DataFrame.from_dict(output_data, orient='index').T
    df = df.fillna("") # Replace NaNs with empty spaces for neatness
    
    output_csv = "cluster_results.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"Cluster outcomes effectively written into: '{output_csv}'\n")
    
    print("=== Cluster Summary ===")
    for cid, users in clusters.items():
        print(f"Cluster {cid}: {', '.join(users)}")

if __name__ == "__main__":
    main()
