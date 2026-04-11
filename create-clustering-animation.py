import os
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import warnings

# Suppress annoying matplotlib backend warnings if running headless
warnings.filterwarnings("ignore", category=UserWarning)

def create_animation(threshold='0.98'):
    results_dir = f"clustering_results_{threshold}"
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        # Fallback list of possible dirs to help users
        dirs = glob.glob("clustering_results_*")
        if dirs:
            print("Available directories:", ", ".join(dirs))
        return
        
    csv_files = glob.glob(os.path.join(results_dir, "cluster_results_after_*.csv"))
    # Sort files alphabetically to match sequential processing step exactly
    csv_files = sorted(csv_files)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}.")
        return
        
    print(f"Generating Network Animation for Threshold {threshold} across {len(csv_files)} sequential steps...")
    
    # Extract chronological history of the system
    history_assignments = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        current_state = {}
        for cluster_col in df.columns:
            # Read columns, clean out NaNs or blanks, map to strings
            users = df[cluster_col].dropna().astype(str).tolist()
            users = [u for u in users if u.strip()]
            if users:
                current_state[cluster_col] = users
                
        # Record the active agent processed this turn
        added_user = os.path.basename(file_path).replace('cluster_results_after_', '').replace('.csv', '')
        history_assignments.append({
            'user_added': added_user,
            'state': current_state
        })
        
    # Setup visualization engine
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1e1e1e')
    fig.tight_layout(pad=3.0)
    
    G = nx.Graph()
    pos = {} # Keep track of position states between frames for smooth transition
    
    # Elegant color mapping mimicking a dynamic tech aesthetic 
    cmap = plt.get_cmap('Set2')
    color_map = {}
    
    def update(frame):
        ax.clear()
        
        # Hold on identical last frame
        if frame >= len(history_assignments):
            frame = len(history_assignments) - 1
            
        step_data = history_assignments[frame]
        state = step_data['state']
        user_added = step_data['user_added']
        
        ax.set_facecolor('#1e1e1e')
        
        # Title UI
        ax.text(0.02, 0.98, f"ALGORITHM VISUALIZATION\nSimilarity Threshold Set: {threshold}", 
                transform=ax.transAxes, fontsize=16, color='white', fontweight='bold', va='top')
        
        ax.text(0.98, 0.98, f"Processing Model:\n{user_added}", 
                transform=ax.transAxes, fontsize=14, color='#00ffcc', fontweight='bold', ha='right', va='top')
        
        ax.axis('off')
        
        # Assemble structural graph
        G.clear()
        current_user_nodes = []
        current_cluster_nodes = []
        
        # Calculate dynamic positions prioritizing cluster cores natively 
        for c_idx, sorted_cluster in enumerate(sorted(state.keys())):
            users = state[sorted_cluster]
            
            if sorted_cluster not in color_map:
                color_map[sorted_cluster] = cmap(c_idx % 8)
                
            # Cluster represent nodes
            G.add_node(sorted_cluster, type='cluster', color=color_map[sorted_cluster])
            current_cluster_nodes.append(sorted_cluster)
            
            # Participant Nodes
            for u in users:
                G.add_node(u, type='user', color=color_map[sorted_cluster])
                G.add_edge(u, sorted_cluster) # Link back to rep
                current_user_nodes.append(u)
                
        # Optimize Layout positions using physics (spring forces) maintaining past anchor locations
        nonlocal pos
        if pos:
            pos = nx.spring_layout(G, pos=pos, k=0.8, iterations=15, seed=42)
        else:
            pos = nx.spring_layout(G, k=0.8, iterations=15, seed=42)
        
        # Draw relationships (edges)
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.4, edge_color='gray', ax=ax)
        
        # Render Representatives (Big Stars)
        nx.draw_networkx_nodes(G, pos, nodelist=current_cluster_nodes, 
                               node_color=[G.nodes[n]['color'] for n in current_cluster_nodes], 
                               node_size=1500, node_shape='*', edgecolors='white', ax=ax)
        
        # Render Participant Models (Circles)
        nx.draw_networkx_nodes(G, pos, nodelist=current_user_nodes, 
                               node_color=[G.nodes[n]['color'] for n in current_user_nodes], 
                               node_size=400, edgecolors='white', alpha=0.9, ax=ax)
                               
        # Render Labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='black',
                                labels={n: n for n in G.nodes()},
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5), ax=ax)
                                
        # Highlight effect over the new arrival
        if user_added in pos:
            ax.plot(*pos[user_added], marker='o', markersize=35, markeredgecolor='#00ffcc', markerfacecolor='none', markeredgewidth=3, zorder=5)
            
        return ax,
        
    num_frames = len(history_assignments) + 5
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=800, blit=False)
    
    out_file = f"clustering_network_{threshold}.gif"
    print(f"Creating animation -> {out_file}...")
    ani.save(out_file, writer='pillow', dpi=100)
    print("Done! Animation successfully rendered.")
    
    # Render and save the final static image cluster structure
    final_img = f"clustering_network_final_{threshold}.png"
    print(f"Saving final state image -> {final_img}...")
    update(len(history_assignments) - 1) # Force axis onto final frame
    fig.savefig(final_img, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"Done! Extracted final image -> {final_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate sequential model clustering via network graphs.")
    parser.add_argument("--threshold", type=str, default="0.98", help="Directory threshold mapping (e.g. 0.90 to 1.0)")
    args = parser.parse_args()
    create_animation(args.threshold)
