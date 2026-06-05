import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Load the REAL 120-matrix data from the final HPC sweep
csv_path = "/Users/kaying/Desktop/summer/combined_results/speedup_data/mast3r_final_accuracy_grid.csv"
save_path = "/Users/kaying/Desktop/summer/latex_package/images/mast3r_all_multipliers_comparison.png"

df = pd.read_csv(csv_path)

def generate_true_120_matrix_plot():
    plt.figure(figsize=(12, 7))
    
    # Define distinct colors for clarity
    colors = {2.5: '#1f77b4', 3.0: '#2ca02c', 3.5: '#ff7f0e', 4.0: '#d62728'}
    
    for m in [2.5, 3.0, 3.5, 4.0]:
        subset = df[df['Multiplier'] == m]
        if subset.empty: continue
        
        plt.plot(subset['N'], subset['Error'], 
                 marker='s', markersize=6, linewidth=2.5,
                 label=f'Bandwidth: {m}x RMT', color=colors[m])

    # Benchmarks and Thresholds
    plt.axhline(y=3.0, color='black', linestyle='--', alpha=0.3, label='3% Consistency Anchor')
    plt.axhline(y=5.0, color='red', linestyle=':', alpha=0.5, label='5% Critical Threshold')
    
    # Highlight the full network resilience
    plt.axvspan(100, 120, color='green', alpha=0.08, label='Full-Network Resilience Zone')

    plt.xlabel('Number of Compressed Matrices (N)', fontsize=13)
    plt.ylabel('Relative 3D Reconstruction Error (%)', fontsize=13)
    plt.title('MASt3R ViT-L: 120-Matrix Fidelity Frontier (HPC Verified)', fontsize=15, pad=20)
    
    plt.grid(True, alpha=0.2, which='both')
    plt.legend(loc='upper left', fontsize=11)
    
    plt.ylim(0, 10) # Focused view on the functional region
    plt.xlim(15, 125)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Regenerated TRUE 120-matrix plot to {save_path}")

if __name__ == "__main__":
    generate_true_120_matrix_plot()
