#!/usr/bin/env python3
"""
Demonstration of the correct clonotype analysis workflow.

This shows the proper order: 
1. Find clonotypes (cell-level clustering)
2. Hierarchically order clonotypes (group-level clustering)
"""

import numpy as np
import pandas as pd
import anndata

def demonstrate_correct_workflow():
    """
    Show the correct order of clonotype discovery and hierarchical ordering.
    """
    
    print("=== Correct Clonotype Analysis Workflow ===\n")
    
    # Create example data
    np.random.seed(42)
    n_cells = 20
    n_variants = 8
    
    # Simulate VAF data with 3 underlying clonotypes
    vaf_matrix = np.random.beta(1, 10, size=(n_cells, n_variants))  # Background noise
    
    # True clonotype 1: cells 0-6, high VAF in variants 0,1
    vaf_matrix[0:7, 0:2] = np.random.beta(3, 1, size=(7, 2)) * 0.8
    
    # True clonotype 2: cells 7-13, high VAF in variants 2,3,4
    vaf_matrix[7:14, 2:5] = np.random.beta(3, 1, size=(7, 3)) * 0.8
    
    # True clonotype 3: cells 14-19, high VAF in variants 5,6,7
    vaf_matrix[14:20, 5:8] = np.random.beta(3, 1, size=(6, 3)) * 0.8
    
    # Create AnnData object
    adata = anndata.AnnData(X=vaf_matrix)
    adata.obs_names = [f"Cell_{i:02d}" for i in range(n_cells)]
    adata.var_names = [f"Variant_{i}" for i in range(n_variants)]
    
    print("1. Original VAF matrix (first 6 cells, all variants):")
    print(pd.DataFrame(vaf_matrix[:6, :], 
                      index=adata.obs_names[:6], 
                      columns=adata.var_names).round(3))
    print()
    
    # STEP 1: Discover clonotypes using individual cell VAF profiles
    print("STEP 1: Discover clonotypes using find_clonotypes()")
    print("        → Uses CELL-LEVEL VAF profiles for clustering")
    print("        → Cosine similarity between individual cells")
    print()
    
    # For demonstration, let's manually assign clonotypes based on our simulation
    # (In real usage, this would be done by find_clonotypes function)
    discovered_clonotypes = (['Clonotype_A'] * 7 + 
                           ['Clonotype_B'] * 7 + 
                           ['Clonotype_C'] * 6)
    
    adata.obs['discovered_clonotypes'] = discovered_clonotypes
    adata.obs['discovered_clonotypes'] = adata.obs['discovered_clonotypes'].astype('category')
    
    print("   Discovered clonotypes:")
    for cell, clono in zip(adata.obs_names, discovered_clonotypes):
        print(f"   {cell}: {clono}")
    print()
    
    # STEP 2: Hierarchically order the discovered clonotypes  
    print("STEP 2: Hierarchically order clonotypes using cluster_clonotypes()")
    print("        → Uses GROUP-LEVEL mean VAF profiles")
    print("        → Cosine similarity between clonotype signatures")
    print()
    
    # Calculate mean VAF profiles for each discovered clonotype
    clonotype_profiles = {}
    unique_clonotypes = adata.obs['discovered_clonotypes'].cat.categories
    
    for clono in unique_clonotypes:
        mask = adata.obs['discovered_clonotypes'] == clono
        cells_in_clono = vaf_matrix[mask, :]
        mean_profile = np.mean(np.sqrt(cells_in_clono), axis=0)  # Square root transformation
        clonotype_profiles[clono] = mean_profile
        
        print(f"   {clono} mean VAF profile (sqrt-transformed):")
        profile_df = pd.DataFrame([mean_profile], 
                                columns=adata.var_names, 
                                index=[clono])
        print(f"   {profile_df.round(3)}")
        print()
    
    # Calculate cosine similarity between clonotype profiles
    from sklearn.metrics.pairwise import cosine_similarity
    
    profile_matrix = np.array([clonotype_profiles[clono] for clono in unique_clonotypes])
    clonotype_similarity = cosine_similarity(profile_matrix)
    
    print("   Cosine similarity between clonotypes:")
    sim_df = pd.DataFrame(clonotype_similarity, 
                         index=unique_clonotypes,
                         columns=unique_clonotypes)
    print(sim_df.round(3))
    print()
    
    # Show hierarchical clustering
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    distance_matrix = 1 - clonotype_similarity
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='complete')
    
    print("   Hierarchical clustering result:")
    print("   → This determines the ORDER of clonotypes in visualizations")
    print("   → More similar clonotypes will be placed closer together")
    print()
    
    print("=== KEY INSIGHT ===")
    print("Step 1: Cell → Cell similarity (discover groups)")
    print("Step 2: Group → Group similarity (order groups)")
    print()
    print("Both use VAF-based cosine similarity, but at different levels!")
    
    return {
        'adata': adata,
        'clonotype_profiles': clonotype_profiles,
        'clonotype_similarity': clonotype_similarity
    }

if __name__ == "__main__":
    results = demonstrate_correct_workflow()
