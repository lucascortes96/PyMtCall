#!/usr/bin/env python3
"""
Demonstration of cosine similarity calculation in clonotype analysis.

This script shows exactly how cosine similarity is calculated in the 
clonotype analysis functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def demonstrate_cosine_similarity():
    """
    Demonstrate how cosine similarity works in clonotype analysis.
    """
    
    print("=== Cosine Similarity in Clonotype Analysis ===\n")
    
    # Create example VAF data
    np.random.seed(42)
    n_cells = 12
    n_variants = 6
    
    # Simulate 3 clonotypes with different VAF patterns
    vaf_matrix = np.zeros((n_cells, n_variants))
    
    # Clonotype 1 (cells 0-3): High VAF in variants 0,1
    vaf_matrix[0:4, 0:2] = np.random.beta(3, 1, size=(4, 2)) * 0.8
    vaf_matrix[0:4, 2:] = np.random.beta(1, 5, size=(4, 4)) * 0.1
    
    # Clonotype 2 (cells 4-7): High VAF in variants 2,3
    vaf_matrix[4:8, 2:4] = np.random.beta(3, 1, size=(4, 2)) * 0.8
    vaf_matrix[4:8, [0,1,4,5]] = np.random.beta(1, 5, size=(4, 4)) * 0.1
    
    # Clonotype 3 (cells 8-11): High VAF in variants 4,5
    vaf_matrix[8:12, 4:6] = np.random.beta(3, 1, size=(4, 2)) * 0.8
    vaf_matrix[8:12, 0:4] = np.random.beta(1, 5, size=(4, 4)) * 0.1
    
    # Add some noise
    vaf_matrix += np.random.normal(0, 0.02, vaf_matrix.shape)
    vaf_matrix = np.clip(vaf_matrix, 0, 1)  # Keep VAF between 0 and 1
    
    print("1. Original VAF Matrix (cells × variants):")
    vaf_df = pd.DataFrame(vaf_matrix, 
                         index=[f"Cell_{i}" for i in range(n_cells)],
                         columns=[f"Var_{i}" for i in range(n_variants)])
    print(vaf_df.round(3))
    print()
    
    # Define clonotype assignments
    clonotypes = ['Clonotype_A'] * 4 + ['Clonotype_B'] * 4 + ['Clonotype_C'] * 4
    
    print("2. Clonotype assignments:")
    for i, (cell, clono) in enumerate(zip(vaf_df.index, clonotypes)):
        print(f"  {cell}: {clono}")
    print()
    
    # Step 1: Calculate group means (like cluster_clonotypes function)
    print("3. Group-wise mean VAF calculation:")
    group_means = []
    unique_clonotypes = ['Clonotype_A', 'Clonotype_B', 'Clonotype_C']
    
    for clono in unique_clonotypes:
        mask = np.array(clonotypes) == clono
        group_vaf = vaf_matrix[mask, :]
        
        # Apply square root transformation and take mean
        mean_vaf = np.mean(np.sqrt(group_vaf), axis=0)
        group_means.append(mean_vaf)
        
        print(f"  {clono}:")
        print(f"    Raw VAF: {np.mean(group_vaf, axis=0).round(3)}")
        print(f"    Sqrt-transformed mean: {mean_vaf.round(3)}")
    
    group_means = np.array(group_means)  # Shape: (3 clonotypes, 6 variants)
    print()
    
    # Step 2: Calculate cosine similarity between clonotypes
    print("4. Cosine similarity between clonotypes:")
    cos_sim = cosine_similarity(group_means)
    
    cos_sim_df = pd.DataFrame(cos_sim, 
                             index=unique_clonotypes,
                             columns=unique_clonotypes)
    print(cos_sim_df.round(3))
    print()
    
    # Step 3: Cell-level cosine similarity (like find_clonotypes function)
    print("5. Cell-level cosine similarity (first 6 cells shown):")
    sqrt_vaf = np.sqrt(vaf_matrix)
    cell_cos_sim = cosine_similarity(sqrt_vaf)
    
    cell_cos_sim_df = pd.DataFrame(cell_cos_sim[:6, :6], 
                                  index=[f"Cell_{i}" for i in range(6)],
                                  columns=[f"Cell_{i}" for i in range(6)])
    print(cell_cos_sim_df.round(3))
    print()
    
    # Interpretation
    print("6. Interpretation:")
    print("   - Cosine similarity ranges from -1 to 1")
    print("   - 1 = identical VAF profiles")
    print("   - 0 = orthogonal VAF profiles") 
    print("   - -1 = opposite VAF profiles")
    print()
    print("   - Group-level similarity compares clonotype VAF signatures")
    print("   - Cell-level similarity compares individual cell VAF profiles")
    print("   - Square root transformation reduces impact of very high VAFs")
    print()
    
    # Show why square root matters
    print("7. Effect of square root transformation:")
    high_vaf = np.array([0.1, 0.5, 0.9])
    print(f"   Original VAFs: {high_vaf}")
    print(f"   After sqrt:    {np.sqrt(high_vaf).round(3)}")
    print("   → Reduces the dominance of high VAF variants")
    
    return {
        'vaf_matrix': vaf_matrix,
        'group_means': group_means,
        'clonotype_similarity': cos_sim,
        'cell_similarity': cell_cos_sim,
        'clonotypes': clonotypes
    }

if __name__ == "__main__":
    results = demonstrate_cosine_similarity()
