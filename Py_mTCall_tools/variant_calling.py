import os
import glob
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import anndata
import warnings

def read_allele_file(path, allele):
    """
    Read a single allele count file and annotate with allele and strand information.

    Args:
        path (str): Path to the allele count file (CSV format).
        allele (str): The allele (A, C, G, or T) corresponding to the file.

    Returns:
        pd.DataFrame: DataFrame with columns for position, cell, forward/reverse counts, and annotated row names.
    """
    df = pd.read_csv(path, sep=",", header=None, names=["pos", "cell", "forw", "rev"])
    df["pos"] = df["pos"].astype(str)
    df["row_forw"] = f"{allele}-" + df["pos"] + "-for"
    df["row_rev"] = f"{allele}-" + df["pos"] + "-rev"
    return df

def build_sparse_matrix(df, cell_lookup, strand):
    """
    Build a sparse matrix for a given strand from a DataFrame of allele counts.

    Args:
        df (pd.DataFrame): DataFrame with allele count data.
        cell_lookup (dict): Mapping from cell barcode to column index.
        strand (str): 'forw' or 'rev' indicating the strand.

    Returns:
        tuple: (coo_matrix, list of row names)
    """
    # strand: 'forw' or 'rev'
    rownames = df[f"row_{strand}"]
    row_lookup = {name: i for i, name in enumerate(rownames.unique())}
    rows = df[f"row_{strand}"].map(row_lookup)
    cols = df["cell"].map(cell_lookup)
    data = df[strand].astype(int)
    mat = coo_matrix((data, (rows, cols)), shape=(len(row_lookup), len(cell_lookup)))
    return mat, list(row_lookup.keys())

def find_allele_files(directory=None):
    """
    Find all allele count files in a directory for alleles A, C, G, and T.

    Args:
        directory (str, optional): Directory to search. Defaults to current working directory.

    Returns:
        dict: Dictionary mapping allele to list of file paths.
    """
    alleles = ['A', 'C', 'G', 'T']
    files = {allele: [] for allele in alleles}
    search_dir = directory if directory is not None else os.getcwd()
    for allele in alleles:
        pattern = os.path.join(search_dir, f"*.{allele}.txt.gz")
        found = glob.glob(pattern)
        files[allele].extend(found)
    return files

def combine_allele_counts(directory=None):
    """
    Combine all allele count files in a directory into a single DataFrame with multi-indexed rows.

    Args:
        directory (str, optional): Directory containing allele count files. Defaults to current working directory.

    Returns:
        pd.DataFrame: DataFrame with multi-index (allele, pos, strand) and columns as cell barcodes.
    """
    alleles = ["A", "C", "G", "T"]
    all_dfs = []
    for allele in alleles:
        pattern = os.path.join(directory if directory else os.getcwd(), f"*.{allele}.txt.gz")
        files = glob.glob(pattern)
        for path in files:
            all_dfs.append(read_allele_file(path, allele))
    all_df = pd.concat(all_dfs, ignore_index=True)
    # Get all unique cell barcodes
    cellbarcodes = all_df["cell"].unique()
    cell_lookup = {cell: i for i, cell in enumerate(cellbarcodes)}
    # Build sparse matrices for forward and reverse
    mat_forw, rownames_forw = build_sparse_matrix(all_df, cell_lookup, "forw")
    mat_rev, rownames_rev = build_sparse_matrix(all_df, cell_lookup, "rev")
    # Stack rownames and matrices
    all_mat = coo_matrix(np.vstack([mat_forw.toarray(), mat_rev.toarray()]))
    all_rownames = rownames_forw + rownames_rev
    # Build MultiIndex from all_rownames
    allele, pos, strand = zip(*(name.split('-', 2) for name in all_rownames))
    multi_index = pd.MultiIndex.from_arrays([allele, pos, strand], names=['allele', 'pos', 'strand'])
    # Build DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(all_mat, index=multi_index, columns=cellbarcodes)
    df = df.reset_index()  # This will make 'allele', 'pos', 'strand' columns
    return df

def read_refallele(directory):
    """
    Find and read the reference allele file in the given directory.

    Args:
        directory (str): Directory containing the refAllele file.

    Returns:
        pd.DataFrame: DataFrame with columns ['pos', 'ref'].
    """
    pattern = os.path.join(directory, '*_refAllele.txt*')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No refAllele file found in {directory} matching '*_refAllele.txt*'")
    refallele_path = files[0]
    return pd.read_csv(refallele_path, sep=None, engine='python', header=None, names=['pos', 'ref'])

def read_coverage(directory):
    """
    Find and read the coverage file in the given directory.

    Args:
        directory (str): Directory containing the coverage file.

    Returns:
        pd.DataFrame: DataFrame with columns ['pos', 'cell', 'coverage'].
    """
    pattern = os.path.join(directory, '*.coverage.txt')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No coverage file found in {directory} matching '*.coverage.txt'")
    coverage_path = files[0]
    df = pd.read_csv(coverage_path, sep=',', header=None, names=['pos', 'cell', 'coverage'])
    df['pos'] = df['pos'].astype(str)
    df['cell'] = df['cell'].astype(str)
    df['coverage'] = pd.to_numeric(df['coverage'], errors='coerce').fillna(0).astype(int)
    print(f"Read coverage file: {df.head()}")
    return df


def identify_variants(counts_df, refallele_df, coverage, low_coverage_threshold=10, stabilize_variance=True, verbose=True, **kwargs):
    """
    Identify variant alleles, compute per-allele and per-cell coverage, and calculate VAF (variant allele fraction) matrices.

    Args:
        counts_df (pd.DataFrame): DataFrame of allele counts (from combine_allele_counts).
        refallele_df (pd.DataFrame): DataFrame of reference alleles (from read_refallele).
        coverage (pd.DataFrame): DataFrame of per-cell, per-position coverage (from read_coverage).
        low_coverage_threshold (int, optional): Minimum coverage to avoid imputation. Defaults to 10.
        stabilize_variance (bool, optional): If True, replace low-coverage VAFs with row means. Defaults to True.
        verbose (bool, optional): If True, print progress messages. Defaults to True.
        **kwargs: Additional arguments (unused).

    Returns:
        tuple: (summary DataFrame, variant count matrix, VAF matrix)
    """
    print("identifying variants")
    # Ensure 'pos' is string in all DataFrames for merging
    counts_df['pos'] = counts_df['pos'].astype(str)
    refallele_df['pos'] = refallele_df['pos'].astype(str)
    # Merge counts_df with refallele_df on 'pos'
    merged = counts_df.merge(refallele_df, on='pos', how='left')
    # Filter where allele != ref
    variants = merged[merged['allele'].str.upper() != merged['ref'].str.upper()]
    meta_cols = ['allele', 'pos', 'strand', 'ref']
    cell_cols = [col for col in variants.columns if col not in meta_cols]
    # Pivot so you have a MultiIndex (allele, pos, strand) and columns are cells
    pivoted = variants.set_index(['allele','pos', 'strand'])[cell_cols]
    pivoted = pivoted.astype(float)
    # Get forward and reverse matrices (MultiIndex: allele, pos, strand; columns: cells)
    fwd_df = pivoted.xs('for', level='strand', drop_level=False)
    rev_df = pivoted.xs('rev', level='strand', drop_level=False)

    # Align indices (should already match, but just in case)
    fwd_df = fwd_df.sort_index()
    rev_df = rev_df.sort_index()

    # Collapse to (allele, pos) index for both
    fwd_mat = fwd_df.droplevel('strand').to_numpy()
    rev_mat = rev_df.droplevel('strand').to_numpy()
    allele_pos_index = fwd_df.index.droplevel('strand')

    # Only use cells where either strand has >0 counts
    mask = (fwd_mat > 0) | (rev_mat > 0)

    # Preallocate result
    strand_concordance = np.full(fwd_mat.shape[0], np.nan)

    # Vectorized correlation calculation
    for i in range(fwd_mat.shape[0]):
        f = fwd_mat[i][mask[i]]
        r = rev_mat[i][mask[i]]
        if f.size > 1:
            # np.corrcoef returns nan if all values are constant, which matches R's behavior
            strand_concordance[i] = np.corrcoef(f, r)[0, 1]

    strand_concordance_df = pd.DataFrame({
        'allele': allele_pos_index.get_level_values('allele'),
        'pos': allele_pos_index.get_level_values('pos'),
        'strand_concordance': strand_concordance
    })
    # For each pos and cell, check if both strands are >2
    mask = pivoted >= 2
    both_strands = mask.groupby(level=['allele', 'pos']).transform('all')  # True only if both for and rev are >2
    # Set values to NaN where not both >2
    filtered = pivoted.where(both_strands)
    # Reset index and merge meta columns back using all unique identifying columns
    filtered = filtered.reset_index()
    filtered = pd.merge(filtered, variants[meta_cols], on=['allele', 'pos', 'strand'], how='left')

    # Now, drop 'strand' and sum cell columns for each (allele, pos)
    cell_cols = [col for col in filtered.columns if col not in meta_cols]
    grouped = (
        filtered
        .drop(columns=['strand'])
        .groupby(['allele', 'pos', 'ref'], as_index=False)[cell_cols]
        .sum(min_count=1)  # min_count=1 keeps NaN if both are NaN
    )
    # Remove rows where all cell columns are NaN
    grouped_nonan = grouped.dropna(how='all', subset=cell_cols)
    # Prepare matrices for VAF calculation
    cell_cols = [col for col in grouped_nonan.columns if col not in ['allele', 'pos', 'ref']]
    # Ensure grouped_nonan index is (allele, pos) for allele-specific info
    grouped_nonan_indexed = grouped_nonan.set_index(['allele', 'pos'])
    var_matrix = grouped_nonan_indexed[cell_cols].to_numpy()
    # --- Build per-cell, per-position coverage matrix from coverage DataFrame ---
    coverage_matrix = coverage.pivot(index='pos', columns='cell', values='coverage').fillna(0)
    coverage_matrix.index = coverage_matrix.index.astype(str)
    coverage_matrix.columns = coverage_matrix.columns.astype(str)
    # For each variant (row in grouped_nonan), get the total coverage for that pos and all cells
    # This will repeat the total coverage for each allele at a given pos, so the shape matches var_matrix (allele, pos, cells)
    pos_list = grouped_nonan_indexed.index.get_level_values('pos')
    cov_matrix = coverage_matrix.loc[pos_list, cell_cols].to_numpy()

    # Compute VAF matrix (variant allele fraction)
    with np.errstate(divide='ignore', invalid='ignore'): 
        vaf_matrix = np.divide(var_matrix, cov_matrix)
        vaf_matrix[~np.isfinite(vaf_matrix)] = 0  # Set NaN/inf to 0

    # Replace low coverage cells with mean (optional, as in R)
    if stabilize_variance:
        low_cov_mask = cov_matrix < low_coverage_threshold
        row_means = np.nanmean(vaf_matrix, axis=1)
        vaf_matrix[low_cov_mask] = np.take(row_means, np.where(low_cov_mask)[0])

    # Compute summary statistics
    variance = np.nanvar(vaf_matrix, axis=1)
    mean = np.nansum(var_matrix, axis=1) / np.nansum(cov_matrix, axis=1)
    confident_mask = ~np.isnan(var_matrix)

    n_cells_conf_detected = np.sum(confident_mask, axis=1)
    n_cells_over_5 = np.sum((vaf_matrix >= 0.05) & confident_mask, axis=1)
    n_cells_over_10 = np.sum((vaf_matrix >= 0.10) & confident_mask, axis=1)
    n_cells_over_50 = np.sum((vaf_matrix >= 0.50) & confident_mask, axis=1)
    mean_coverage = np.nanmean(cov_matrix, axis=1)
    grouped_nonan = pd.merge(grouped_nonan, strand_concordance_df, on=['allele', 'pos'], how='left')

    # Add summary columns to your DataFrame
    grouped_nonan['mean'] = mean
    grouped_nonan['variance'] = variance
    grouped_nonan['n_cells_conf_detected'] = n_cells_conf_detected
    grouped_nonan['n_cells_over_5'] = n_cells_over_5
    grouped_nonan['n_cells_over_10'] = n_cells_over_10
    grouped_nonan['n_cells_over_50'] = n_cells_over_50
    grouped_nonan['mean_coverage'] = mean_coverage

    return grouped_nonan, var_matrix, vaf_matrix

def process_and_integrate_variants(input_folder, adata, vaf_layer_name="vaf", summary_prefix="variant_"):
    """
    Run the full variant calling workflow and integrate results into an AnnData object.

    Args:
        input_folder (str): Directory with allele count, refAllele, and coverage files.
        adata (anndata.AnnData): AnnData object with cell barcodes in adata.obs.index.
        vaf_layer_name (str, optional): Name for the VAF matrix layer in AnnData. Defaults to "vaf".
        summary_prefix (str, optional): Prefix for summary columns in adata.var. Defaults to "variant_".

    Returns:
        anndata.AnnData: New AnnData object with integrated variant and VAF information.
    """
    # Step 1: Run all processing steps
    counts_df = combine_allele_counts(input_folder)
    refallele_df = read_refallele(input_folder)
    coverage = read_coverage(input_folder)
    summary_df, var_matrix, vaf_matrix_full = identify_variants(
        counts_df=counts_df,
        refallele_df=refallele_df,
        coverage=coverage,
        low_coverage_threshold=10,
        stabilize_variance=True,
        verbose=True
    )

    # Step 2: Extract VAF matrix and summary stats
    # Identify cell barcodes present in both vaf_matrix_full and AnnData
    cell_barcodes = adata.obs.index.astype(str)
    # Use the same cell_cols as in identify_variants
    cell_cols = list(summary_df.columns.difference(["allele", "pos", "ref", "mean", "variance", "n_cells_conf_detected", "n_cells_over_5", "n_cells_over_10", "n_cells_over_50", "mean_coverage", "strand_concordance"]))
    matching_cells = [c for c in cell_barcodes if c in cell_cols]
    if not matching_cells:
        raise ValueError("No matching cell barcodes between VAF matrix and AnnData object.")
    # Find the indices of these barcodes in cell_cols
    cell_indices = [cell_cols.index(cb) for cb in matching_cells]
    # Subset vaf_matrix_full to only those cells, in the correct order
    vaf_matrix = vaf_matrix_full[:, cell_indices]
    # Variants (rows of summary_df)
    variant_index = summary_df[["allele", "pos", "ref"]].astype(str).agg("-".join, axis=1)
    # Build new AnnData object with correct shape (cells, variants)
    adata_new = anndata.AnnData(
        X=np.zeros((len(matching_cells), len(summary_df))),  # placeholder
        obs=adata.obs.loc[matching_cells].copy(),
        var=summary_df[["mean", "variance", "n_cells_conf_detected", "n_cells_over_5", "n_cells_over_10", "n_cells_over_50", "mean_coverage", "strand_concordance"]].copy()
    )
    # KEEPING UMAP EMBEDDING, not automatic 
    if "X_umap" in adata.obsm:
        # Subset and align UMAP to the new cell order
        umap_df = pd.DataFrame(adata.obsm["X_umap"], index=adata.obs.index)
        adata_new.obsm["X_umap"] = umap_df.loc[adata_new.obs.index].to_numpy()
        adata_new.var.index = variant_index
    else:
        warnings.warn("No UMAP embedding ('X_umap') found in input AnnData. Downstream UMAP plots will not be available.")
        adata_new.var.index = variant_index
    # Set VAF matrix as a layer (cells, variants)
    vaf_matrix_cells_variants = vaf_matrix.T  # (cells, variants)
    adata_new.layers[vaf_layer_name] = vaf_matrix_cells_variants
    # Efficiently add per-variant count and VAF columns to .obs
    # Use matching_cells for .obs columns
    var_matrix_cells = var_matrix.T[cell_indices, :]
    count_df = pd.DataFrame(var_matrix_cells, index=adata_new.obs.index, columns=[f"count_{v}" for v in variant_index])
    vaf_df = pd.DataFrame(vaf_matrix_cells_variants, index=adata_new.obs.index, columns=[f"vaf_{v}" for v in variant_index])
    adata_new.obs = pd.concat([adata_new.obs, count_df, vaf_df], axis=1)
    # Optionally, add a list of detected mutations per cell (where VAF > 0)
    detected_mask = vaf_matrix_cells_variants > 0
    adata_new.obs["mutations"] = [list(variant_index[detected_mask[i]]) for i in range(detected_mask.shape[0])]
    return adata_new