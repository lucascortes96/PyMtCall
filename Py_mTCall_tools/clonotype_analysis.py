import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings

def cluster_clonotypes(adata, layer=None, group_by=None):
    """
    Hierarchically cluster EXISTING clonotypes/groups based on cosine similarity of their mean allele frequencies.
    
    This function is used AFTER clonotypes have been identified to:
    1. Order clonotypes by similarity for visualization
    2. Order features/variants by their discriminative power
    
    This is equivalent to the R function ClusterClonotypes.
    
    Args:
        adata (anndata.AnnData): AnnData object containing allele frequency data
        layer (str, optional): Layer to use for clustering. If None, uses adata.X
        group_by (str, optional): Column in adata.obs with EXISTING group/clonotype assignments. 
                                 If None, uses current clustering (leiden/louvain)
    
    Returns:
        dict: Dictionary with 'cells' and 'features' hierarchical clustering results
            - 'cells': hierarchical clustering of groups/clonotypes
            - 'features': hierarchical clustering of variants/features
            - 'group_means': mean VAF profiles for each group
            - 'groups': group names in original order
    """
    
    # Store original grouping
    if group_by is None:
        if 'leiden' in adata.obs.columns:
            group_col = 'leiden'
        elif 'louvain' in adata.obs.columns:
            group_col = 'louvain'
        else:
            # Create a single group if no clustering exists
            adata.obs['temp_group'] = 'group_0'
            group_col = 'temp_group'
    else:
        group_col = group_by
    
    # Get the data matrix
    if layer is not None:
        if layer in adata.layers:
            mat = adata.layers[layer]
        else:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
    else:
        mat = adata.X
    
    # Convert to dense if sparse
    if hasattr(mat, 'toarray'):
        mat = mat.toarray()
    
    # Find mean allele frequency of each variant in each group
    unique_groups = adata.obs[group_col].unique()
    
    # Initialize matrix to store group means
    group_means = np.zeros((mat.shape[1], len(unique_groups)))  # features x groups
    
    for i, group in enumerate(unique_groups):
        group_mask = adata.obs[group_col] == group
        group_cells = mat[group_mask, :]
        
        if group_cells.shape[0] > 0:
            # Take square root of positive values only (for VAF data this should be non-negative)
            # For general data, take absolute value first to avoid NaN
            group_cells_positive = np.abs(group_cells)
            group_means[:, i] = np.mean(np.sqrt(group_cells_positive), axis=0)
    
    # Clean up temporary column if created
    if group_by is None and 'temp_group' in adata.obs.columns:
        adata.obs.drop('temp_group', axis=1, inplace=True)
    
    # Calculate cosine similarity matrices
    # Replace any remaining NaN or inf values
    group_means = np.nan_to_num(group_means, nan=0.0, posinf=0.0, neginf=0.0)
    
    # For groups (cells in R terminology)
    cos_similarity = cosine_similarity(group_means.T)  # groups x groups
    
    # For features  
    cos_similarity_features = cosine_similarity(group_means)  # features x features
    
    # Replace NaN with 0 (equivalent to R code)
    cos_similarity[np.isnan(cos_similarity)] = 0
    cos_similarity_features[np.isnan(cos_similarity_features)] = 0
    
    # Convert similarity to distance for hierarchical clustering
    cos_distance = 1 - cos_similarity
    cos_distance_features = 1 - cos_similarity_features
    
    # Ensure diagonal is 0 (sometimes floating point errors)
    np.fill_diagonal(cos_distance, 0)
    np.fill_diagonal(cos_distance_features, 0)
    
    # Perform hierarchical clustering
    # Convert distance matrices to condensed form for linkage
    condensed_dist = squareform(cos_distance)
    condensed_dist_features = squareform(cos_distance_features)
    
    # Hierarchical clustering
    hc_cells = linkage(condensed_dist, method='complete')
    hc_features = linkage(condensed_dist_features, method='complete')
    
    return {
        'cells': hc_cells,
        'features': hc_features,
        'group_means': group_means,
        'groups': unique_groups
    }

def find_clonotypes(adata, layer=None, features=None, metric='cosine', 
                   resolution=1.0, k=10, algorithm='leiden'):
    """
    Identify groups of related cells from allele frequency data.
    
    This function is equivalent to the R function FindClonotypes.
    
    Args:
        adata (anndata.AnnData): AnnData object containing allele frequency data
        layer (str, optional): Layer to use for analysis. If None, uses adata.X
        features (list, optional): Features to include when constructing neighbor graph. 
                                 If None, uses all features
        metric (str): Distance metric to use for neighbor graph ('cosine', 'euclidean', etc.)
        resolution (float): Clustering resolution to use
        k (int): Number of neighbors for neighbor graph construction
        algorithm (str): Community detection algorithm ('leiden' or 'louvain')
    
    Returns:
        anndata.AnnData: Updated AnnData object with clonotype assignments
    """
    
    # Get the data matrix
    if layer is not None:
        if layer in adata.layers:
            mat = adata.layers[layer].copy()
        else:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
    else:
        mat = adata.X.copy()
    
    # Convert to dense if sparse
    if hasattr(mat, 'toarray'):
        mat = mat.toarray()
    
    # Select features
    if features is not None:
        feature_indices = [i for i, f in enumerate(adata.var_names) if f in features]
        mat = mat[:, feature_indices]
        selected_features = [adata.var_names[i] for i in feature_indices]
    else:
        selected_features = list(adata.var_names)
    
    # Take square root and transpose (equivalent to R: sqrt(t(mat)))
    mat_transformed = np.sqrt(mat)
    
    # Create a temporary AnnData object for clustering
    adata_temp = anndata.AnnData(X=mat_transformed)
    adata_temp.obs_names = adata.obs_names
    adata_temp.var_names = selected_features
    
    # Construct neighbor graph using scanpy
    try:
        sc.pp.neighbors(adata_temp, n_neighbors=k, metric=metric, use_rep='X')
    except Exception as e:
        warnings.warn(f"Error in neighbor graph construction: {e}. Using simpler approach.")
        # Fallback to direct distance-based clustering
        from sklearn.cluster import KMeans
        n_clusters = max(2, min(10, int(np.sqrt(adata_temp.n_obs // 2))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(mat_transformed)
        adata_temp.obs[f'kmeans'] = [f'cluster_{i}' for i in cluster_labels]
        adata_temp.obs[f'kmeans'] = adata_temp.obs[f'kmeans'].astype('category')
        cluster_col = 'kmeans'
        
        # Create dummy neighbor graph matrices
        n_cells = adata_temp.n_obs
        from scipy.sparse import csr_matrix
        adata_temp.obsp['connectivities'] = csr_matrix((n_cells, n_cells))
        adata_temp.obsp['distances'] = csr_matrix((n_cells, n_cells))
        
        # Skip the clustering step since we already did it
        clustering_done = True
    else:
        clustering_done = False
    
    # Perform clustering (if not already done in fallback)
    if not clustering_done:
        try:
            if algorithm.lower() == 'leiden':
                sc.tl.leiden(adata_temp, resolution=resolution)
                cluster_col = 'leiden'
            elif algorithm.lower() == 'louvain':
                sc.tl.louvain(adata_temp, resolution=resolution)
                cluster_col = 'louvain'
            else:
                raise ValueError("algorithm must be 'leiden' or 'louvain'")
        except ImportError as e:
            if 'igraph' in str(e):
                warnings.warn("igraph not available, falling back to simple k-means clustering")
                from sklearn.cluster import KMeans
                n_clusters = max(2, min(10, int(np.sqrt(adata_temp.n_obs // 2))))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(mat_transformed)
                adata_temp.obs[f'kmeans'] = [f'cluster_{i}' for i in cluster_labels]
                adata_temp.obs[f'kmeans'] = adata_temp.obs[f'kmeans'].astype('category')
                cluster_col = 'kmeans'
            else:
                raise e
    
    # Transfer clustering results to original adata
    adata.obs[f'clonotype_{algorithm}'] = adata_temp.obs[cluster_col].astype('category')
    
    # Transfer neighbor graph information
    adata.obsp[f'clonotype_connectivities'] = adata_temp.obsp['connectivities']
    adata.obsp[f'clonotype_distances'] = adata_temp.obsp['distances']
    
    # ===== IMPORTANT: Two-step process =====
    # Step 1: Already completed above - initial clustering to discover clonotypes
    # Step 2: Now reorder/reorganize the discovered clonotypes using hierarchical clustering
    
    # Perform hierarchical clustering to reorder levels
    hc_results = cluster_clonotypes(adata, layer=layer, group_by=f'clonotype_{algorithm}')
    
    # Reorder cluster levels based on hierarchical clustering
    from scipy.cluster.hierarchy import leaves_list
    cell_order = leaves_list(hc_results['cells'])
    feature_order = leaves_list(hc_results['features'])
    
    # Get the groups in the new order
    ordered_groups = [hc_results['groups'][i] for i in cell_order]
    
    # Create mapping from old to new cluster names
    old_clusters = adata.obs[f'clonotype_{algorithm}'].cat.categories
    cluster_mapping = {old_clusters[i]: f'clonotype_{j}' 
                      for j, group in enumerate(ordered_groups) 
                      for i, old_cluster in enumerate(old_clusters) 
                      if str(old_cluster) == str(group)}
    
    # Apply the mapping
    adata.obs[f'clonotype_{algorithm}'] = adata.obs[f'clonotype_{algorithm}'].map(cluster_mapping).astype('category')
    
    # Set variable features based on hierarchical clustering order
    if features is None:
        ordered_features = [adata.var_names[i] for i in feature_order]
        adata.var['highly_variable'] = False
        adata.var.loc[ordered_features, 'highly_variable'] = True
        # Store the feature order
        adata.uns['clonotype_feature_order'] = ordered_features
    
    # Store clustering parameters and results
    adata.uns['clonotype_params'] = {
        'layer': layer,
        'features': features,
        'metric': metric,
        'resolution': resolution,
        'k': k,
        'algorithm': algorithm
    }
    
    adata.uns['clonotype_hierarchy'] = hc_results
    
    return adata

def plot_clonotype_heatmap(adata, layer=None, group_by=None, features=None, 
                          figsize=(10, 8), cmap='viridis', save=None):
    """
    Plot a heatmap of clonotype allele frequencies.
    
    Args:
        adata (anndata.AnnData): AnnData object with clonotype assignments
        layer (str, optional): Layer to use for plotting. If None, uses adata.X
        group_by (str, optional): Column in adata.obs for grouping. If None, uses clonotype assignments
        features (list, optional): Features to include in heatmap. If None, uses all features
        figsize (tuple): Figure size
        cmap (str): Colormap for heatmap
        save (str, optional): Path to save figure
    
    Returns:
        matplotlib figure
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get clustering results
    hc_results = cluster_clonotypes(adata, layer=layer, group_by=group_by)
    
    # Get feature and group orders from hierarchical clustering
    from scipy.cluster.hierarchy import leaves_list
    feature_order = leaves_list(hc_results['features'])
    group_order = leaves_list(hc_results['cells'])
    
    # Reorder the group means matrix
    ordered_means = hc_results['group_means'][feature_order, :][:, group_order]
    
    # Create labels
    if features is not None:
        feature_labels = [f for f in features if f in adata.var_names]
    else:
        feature_labels = list(adata.var_names)
    
    ordered_feature_labels = [feature_labels[i] for i in feature_order]
    ordered_group_labels = [str(hc_results['groups'][i]) for i in group_order]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(ordered_means, 
                xticklabels=ordered_group_labels,
                yticklabels=ordered_feature_labels,
                cmap=cmap, 
                ax=ax,
                cbar_kws={'label': 'Mean Allele Frequency'})
    
    ax.set_xlabel('Clonotypes')
    ax.set_ylabel('Variants')
    ax.set_title('Clonotype Allele Frequency Heatmap')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_clonotypes_from_variants(adata, vaf_layer_name="vaf", min_cells=10, 
                                   min_vaf=0.1, min_coverage=20, min_strand_concordance=0.3,
                                   max_variance=0.1, resolution=1.0, k=10):
    """
    Perform clonotype analysis using only confidently detected variants.
    
    This function integrates with the process_and_integrate_variants output
    to perform clonotype analysis based on high-confidence variant allele frequencies.
    
    Args:
        adata (anndata.AnnData): AnnData object from process_and_integrate_variants
        vaf_layer_name (str): Name of the VAF layer (default: "vaf")
        min_cells (int): Minimum number of cells with confident variant detection
        min_vaf (float): Minimum VAF threshold for confident detection (default: 10%)
        min_coverage (float): Minimum mean coverage depth for a variant
        min_strand_concordance (float): Minimum strand concordance for confident detection
        max_variance (float): Maximum VAF variance across cells (filters noisy variants)
        resolution (float): Clustering resolution
        k (int): Number of neighbors for graph construction
    
    Returns:
        anndata.AnnData: Updated AnnData object with clonotype assignments
    """
    
    # Check if we have the VAF matrix from variant calling
    vaf_matrix_key = f'{vaf_layer_name}_matrix'
    if vaf_matrix_key not in adata.obsm:
        raise ValueError(f"VAF matrix '{vaf_matrix_key}' not found in adata.obsm. "
                        "Please run process_and_integrate_variants first.")
    
    # Get VAF matrix
    vaf_matrix = adata.obsm[vaf_matrix_key]
    
    # Filter variants based on confidence criteria
    if 'variant_info' in adata.uns:
        variant_info = adata.uns['variant_info']
        variant_names = adata.uns['variant_names']
        
        # Apply multiple confidence filters
        confidence_filters = {
            'min_cells_confident': variant_info['n_cells_conf_detected'] >= min_cells,
            'min_cells_high_vaf': variant_info['n_cells_over_5'] >= min_cells,
            'sufficient_coverage': variant_info['mean_coverage'] >= min_coverage,
            'strand_concordance': (variant_info['strand_concordance'] >= min_strand_concordance) | 
                                 variant_info['strand_concordance'].isna(),  # Accept NaN (single-strand variants)
            'low_variance': variant_info['variance'] <= max_variance
        }
        
        # Combine all filters
        confident_variants = pd.Series(True, index=variant_info.index)
        for filter_name, filter_mask in confidence_filters.items():
            confident_variants = confident_variants & filter_mask
            print(f"After {filter_name}: {confident_variants.sum()} variants remaining")
        
        # Get selected variants
        selected_variant_indices = confident_variants[confident_variants].index
        # The indices are actually the variant names themselves, not positions
        selected_variants = list(selected_variant_indices)
        
        # Filter VAF matrix using boolean mask
        vaf_matrix_filtered = vaf_matrix[:, confident_variants]
        
        print(f"\nFinal selection: Using {len(selected_variants)} high-confidence variants for clonotype analysis")
        print(f"Filtering criteria applied:")
        print(f"  - Confidently detected in >= {min_cells} cells")
        print(f"  - VAF >= {min_vaf:.1%} in >= {min_cells} cells") 
        print(f"  - Mean coverage >= {min_coverage}x")
        print(f"  - Strand concordance >= {min_strand_concordance} (or single-strand)")
        print(f"  - VAF variance <= {max_variance}")
        
    else:
        # Use all variants if no filtering info available (fallback)
        vaf_matrix_filtered = vaf_matrix
        selected_variants = [f"variant_{i}" for i in range(vaf_matrix.shape[1])]
        
        print(f"Warning: No variant confidence info found. Using all {vaf_matrix.shape[1]} variants for clonotype analysis")
    
    # Create temporary AnnData object with VAF data
    adata_vaf = anndata.AnnData(X=vaf_matrix_filtered)
    adata_vaf.obs_names = adata.obs_names
    adata_vaf.var_names = selected_variants
    
    # Perform clonotype analysis
    adata_vaf = find_clonotypes(
        adata_vaf, 
        layer=None,  # Use X directly
        features=None,  # Use all selected variants
        resolution=resolution,
        k=k,
        algorithm='leiden'
    )
    
    # Transfer results back to original adata
    clonotype_col = 'clonotype_leiden'
    adata.obs[clonotype_col] = adata_vaf.obs[clonotype_col]
    
    # Transfer neighbor graph
    adata.obsp['clonotype_connectivities'] = adata_vaf.obsp['clonotype_connectivities']
    adata.obsp['clonotype_distances'] = adata_vaf.obsp['clonotype_distances']
    
    # Store clonotype-specific information
    adata.uns['clonotype_analysis'] = {
        'selected_variants': selected_variants,
        'n_variants_used': len(selected_variants),
        'min_cells_threshold': min_cells,
        'min_vaf_threshold': min_vaf,
        'hierarchy': adata_vaf.uns.get('clonotype_hierarchy', None)
    }
    
    # Calculate clonotype statistics
    clonotype_stats = []
    for clonotype in adata.obs[clonotype_col].cat.categories:
        clonotype_mask = adata.obs[clonotype_col] == clonotype
        n_cells = clonotype_mask.sum()
        
        # Get variants characteristic of this clonotype
        clonotype_vaf = vaf_matrix_filtered[clonotype_mask, :]
        mean_vaf = np.mean(clonotype_vaf, axis=0)
        
        # Find variants with high mean VAF in this clonotype
        high_vaf_variants = np.where(mean_vaf > min_vaf)[0]
        characteristic_variants = [selected_variants[i] for i in high_vaf_variants]
        
        clonotype_stats.append({
            'clonotype': clonotype,
            'n_cells': n_cells,
            'n_characteristic_variants': len(characteristic_variants),
            'characteristic_variants': characteristic_variants,
            'mean_vaf_all_variants': np.mean(mean_vaf)
        })
    
    adata.uns['clonotype_stats'] = pd.DataFrame(clonotype_stats)
    
    print(f"Identified {len(adata.obs[clonotype_col].cat.categories)} clonotypes")
    print(f"Clonotype sizes: {adata.obs[clonotype_col].value_counts().to_dict()}")
    
    return adata

def plot_clonotype_vaf_heatmap(adata, vaf_layer_name="vaf", clonotype_col="clonotype_leiden",
                              top_variants=50, figsize=(12, 10), save=None):
    """
    Plot VAF heatmap for clonotypes using variant data.
    
    Args:
        adata (anndata.AnnData): AnnData object with clonotype assignments
        vaf_layer_name (str): Name of the VAF layer
        clonotype_col (str): Column name for clonotype assignments
        top_variants (int): Number of top variants to show
        figsize (tuple): Figure size
        save (str, optional): Path to save figure
    
    Returns:
        matplotlib figure
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get VAF matrix
    vaf_matrix_key = f'{vaf_layer_name}_matrix'
    if vaf_matrix_key not in adata.obsm:
        raise ValueError(f"VAF matrix '{vaf_matrix_key}' not found")
    
    vaf_matrix = adata.obsm[vaf_matrix_key]
    
    # Get variant names
    if 'variant_names' in adata.uns:
        variant_names = adata.uns['variant_names']
    else:
        variant_names = [f"variant_{i}" for i in range(vaf_matrix.shape[1])]
    
    # Calculate mean VAF per clonotype
    clonotype_means = []
    clonotype_labels = []
    
    for clonotype in adata.obs[clonotype_col].cat.categories:
        clonotype_mask = adata.obs[clonotype_col] == clonotype
        mean_vaf = np.mean(vaf_matrix[clonotype_mask, :], axis=0)
        clonotype_means.append(mean_vaf)
        clonotype_labels.append(f"{clonotype} (n={clonotype_mask.sum()})")
    
    clonotype_means = np.array(clonotype_means)
    
    # Select top variants by variance across clonotypes
    variant_variance = np.var(clonotype_means, axis=0)
    top_variant_indices = np.argsort(variant_variance)[-top_variants:]
    
    # Create heatmap data
    heatmap_data = clonotype_means[:, top_variant_indices]
    selected_variant_names = [variant_names[i] for i in top_variant_indices]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(heatmap_data.T,
                xticklabels=clonotype_labels,
                yticklabels=selected_variant_names,
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Mean VAF'})
    
    ax.set_xlabel('Clonotypes')
    ax.set_ylabel('Variants')
    ax.set_title(f'Clonotype VAF Heatmap (Top {top_variants} variants by variance)')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    return fig

# Utility function for setting default values (equivalent to R's SetIfNull)
def set_if_null(x, y):
    """Set x to y if x is None, otherwise return x."""
    return y if x is None else x

# Example usage of clonotype analysis functions
def example_clonotype_analysis():
    """
    Example showing how to use the clonotype analysis functions
    with the variant calling workflow.
    """
    
    # This assumes you have already run:
    # 1. process_and_integrate_variants() to get variant data
    # 2. Have an AnnData object with VAF information
    
    # Import the clonotype analysis functions
    from Py_mTCall_tools import (
        analyze_clonotypes_from_variants,
        plot_clonotype_vaf_heatmap,
        cluster_clonotypes,
        find_clonotypes
    )
    
    # Example workflow:
    
    # Step 1: Run variant calling (assuming this is already done)
    # input_folder = "path/to/variant/files"
    # adata = process_and_integrate_variants(input_folder, adata)
    
    # Step 2: Analyze clonotypes based on variant allele frequencies
    # adata = analyze_clonotypes_from_variants(
    #     adata, 
    #     vaf_layer_name="vaf",
    #     min_cells=5,          # Require variant in at least 5 cells
    #     min_vaf=0.05,         # Minimum 5% VAF
    #     resolution=1.0,       # Clustering resolution
    #     k=10                  # Number of neighbors
    # )
    
    # Step 3: Visualize clonotype VAF patterns
    # fig = plot_clonotype_vaf_heatmap(
    #     adata,
    #     vaf_layer_name="vaf",
    #     clonotype_col="clonotype_leiden",
    #     top_variants=50,
    #     figsize=(12, 10),
    #     save="clonotype_vaf_heatmap.pdf"
    # )
    
    # Step 4: Access clonotype statistics
    # clonotype_stats = adata.uns['clonotype_stats']
    # print("Clonotype statistics:")
    # print(clonotype_stats)
    
    # Step 5: Optional - manual clustering with custom parameters
    # For more control, you can use the lower-level functions:
    # hc_results = cluster_clonotypes(adata, layer="vaf_matrix", group_by="clonotype_leiden")
    # adata = find_clonotypes(adata, layer="vaf_matrix", resolution=0.5, k=15)
    
    print("Example clonotype analysis workflow completed!")
    print("Key functions:")
    print("- analyze_clonotypes_from_variants(): Main function for clonotype analysis")
    print("- plot_clonotype_vaf_heatmap(): Visualize VAF patterns across clonotypes") 
    print("- cluster_clonotypes(): Hierarchical clustering of clonotypes")
    print("- find_clonotypes(): Graph-based clustering with neighbor graph")

# Note: To run the example, call example_clonotype_analysis() manually
# example_clonotype_analysis()

def get_variant_confidence_summary(adata, vaf_layer_name="vaf"):
    """
    Get a summary of variant confidence metrics for quality assessment.
    
    Args:
        adata (anndata.AnnData): AnnData object with variant data
        vaf_layer_name (str): Name of the VAF layer
        
    Returns:
        pd.DataFrame: Summary of variant confidence metrics
    """
    
    if 'variant_info' not in adata.uns:
        print("No variant confidence info available")
        return None
    
    variant_info = adata.uns['variant_info']
    
    summary = pd.DataFrame({
        'metric': [
            'Total variants',
            'Mean cells confident detection',
            'Mean cells VAF >5%',
            'Mean cells VAF >10%', 
            'Mean cells VAF >50%',
            'Mean coverage depth',
            'Mean strand concordance',
            'Mean VAF variance'
        ],
        'value': [
            len(variant_info),
            variant_info['n_cells_conf_detected'].mean(),
            variant_info['n_cells_over_5'].mean(),
            variant_info['n_cells_over_10'].mean(),
            variant_info['n_cells_over_50'].mean(), 
            variant_info['mean_coverage'].mean(),
            variant_info['strand_concordance'].mean(),
            variant_info['variance'].mean()
        ]
    })
    
    return summary