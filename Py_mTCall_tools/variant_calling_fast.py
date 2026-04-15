"""
Fast Variant Calling - 
-Takes: 
    - adata object and input folder with allele counts, reference alleles, and coverage files.

-Returns:
    - adata object with mitochondrial variant information added.  
    - VARIANT LEVEL SUMMARY STATISTICS are stored in adata.uns['variant_summary']
        - This includes variance, number of cells confidently detected, mean coverage and overall VAF
        - Overall VAF is important as it contains the most accurate calculation of the VAF in a single variant 
    - MATRIX LEVEL (cell, call) INFORMATION is stored in adata.obsm and includes:
        - variant vaf
        - variant counts
        - variant coverage

"""

"""
List of 'fake' mitochondrial variants as per the brilliant Prof Gavin Hudson:

3107N>C
301:302
309:311
316
3107:3109
16182:16183

"""

import os
import glob
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import anndata
import gc


def read_allele_file(path, allele):
    """Read a single allele count file and annotate with allele and strand information."""
    df = pd.read_csv(path, sep=",", header=None, names=["pos", "cell", "forw", "rev"], on_bad_lines="skip")
    df['row_forw'] = (allele + '-' + df['pos'].astype(str) + '-for').astype('category')
    df['row_rev'] = (allele + '-' + df['pos'].astype(str) + '-rev').astype('category')
    return df

##New sparse matrix build used to handle categorical types 
def build_sparse_matrix(df, cell_lookup, strand):
    """Build a sparse matrix for a given strand from a DataFrame of allele counts."""
    rownames = df[f"row_{strand}"].astype(str)
    row_lookup = {name: i for i, name in enumerate(rownames.unique())}
    rows = rownames.map(row_lookup)
    cols = df["cell"].map(cell_lookup)
    data = df[strand].astype(int)
    mat = coo_matrix((data, (rows, cols)), shape=(len(row_lookup), len(cell_lookup)))
    return mat, list(row_lookup.keys())


def combine_allele_counts(directory=None):
    """
    Combine all allele count files - FIXED VERSION that prevents 0-count variants.
    """
    alleles = ["A", "C", "G", "T"]
    all_dfs = []
    
    for allele in alleles:
        # Look for both compressed and uncompressed files
        pattern_gz = os.path.join(directory if directory else os.getcwd(), f"*.{allele}.txt.gz")
        pattern_txt = os.path.join(directory if directory else os.getcwd(), f"*.{allele}.txt")
        files = glob.glob(pattern_gz) + glob.glob(pattern_txt)
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
    
    # Build DataFrame and IMMEDIATELY filter out 0-count rows (BUG FIX)
    df = pd.DataFrame.sparse.from_spmatrix(all_mat, index=multi_index, columns=cellbarcodes)
    
    # Convert to dense for filtering (this is the key fix)
    df_dense = df.sparse.to_dense()
    non_zero_rows = (df_dense != 0).any(axis=1)
    df_filtered = df_dense[non_zero_rows]
    
    # Convert back to the expected format
    df_filtered = df_filtered.reset_index()  # This makes 'allele', 'pos', 'strand' columns
    return df_filtered


def read_refallele(directory):
    """Find and read the reference allele file."""
    pattern = os.path.join(directory, '*_refAllele.txt*')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No refAllele file found in {directory}")
    refallele_path = files[0]
    return pd.read_csv(refallele_path, sep=None, engine='python', header=None, names=['pos', 'ref'])


def read_coverage(directory, variant_positions=None):
    """Find and read the coverage file, optionally filtering to specific positions.
    REMOVE fake variants
    """
    fake_variants = ['3107', '301', '302', '309', '310', '311', '316', '3107', '3108', '3109', '16182', '16183']
    pattern = os.path.join(directory, '*.coverage.txt')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No coverage file found in {directory}")
    coverage_path = files[0]
    
    if variant_positions is not None:
        # Read in chunks and filter to reduce memory usage
        chunk_list = []
        chunk_size = 1000000  # 1M rows at a time
        
        for chunk in pd.read_csv(coverage_path, sep=',', header=None, names=['pos', 'cell', 'coverage'], chunksize=chunk_size):
            chunk['pos'] = chunk['pos'].astype(str)
            # Filter to only positions we need and remove fake variants
            filtered_chunk = chunk[chunk['pos'].isin(variant_positions) & ~chunk['pos'].isin(fake_variants)]
            if len(filtered_chunk) > 0:
                chunk_list.append(filtered_chunk)
        
        if chunk_list:
            df = pd.concat(chunk_list, ignore_index=True)
        else:
            # No matching positions found, return empty DataFrame with correct structure
            df = pd.DataFrame(columns=['pos', 'cell', 'coverage'])
            df['pos'] = df['pos'].astype(str)
    else:
        # Original behavior - read everything
        df = pd.read_csv(coverage_path, sep=',', header=None, names=['pos', 'cell', 'coverage'])
        df['pos'] = df['pos'].astype(str)
    
    df['cell'] = df['cell'].astype(str)
    df['coverage'] = pd.to_numeric(df['coverage'], errors='coerce').fillna(0).astype(int)
    return df


def identify_variants_fast(counts_df, refallele_df, coverage, min_strand_count=2, verbose=True, chunk_size=1000, coverage_threshold=None, **kwargs):
    """
    Memory-optimized variant identification - processes variants in chunks to avoid memory explosion.
    """
    fake_variants = ['3107', '301', '302', '309', '310', '311', '316', '3107', '3108', '3109', '16182', '16183']
    if verbose:
        print("Identifying variants (memory-optimized chunked method)...")
    
    # Ensure 'pos' is string in all DataFrames for merging
    counts_df['pos'] = counts_df['pos'].astype(str)
    refallele_df['pos'] = refallele_df['pos'].astype(str)
    # Remove fake variants from refallele_df
    refallele_df = refallele_df[~refallele_df['pos'].isin(fake_variants)]
    
    # Merge counts_df with refallele_df on 'pos'
    merged = counts_df.merge(refallele_df, on='pos', how='left')
    # Drop rows with missing reference to avoid misclassifying as variants
    merged = merged.dropna(subset=['ref'])
    # Filter where allele != ref (this should now have NO 0-count entries)
    variants = merged[merged['allele'].str.upper() != merged['ref'].str.upper()]
    meta_cols = ['allele', 'pos', 'strand', 'ref']
    cell_cols = [col for col in variants.columns if col not in meta_cols]
    
    
    

    
    # NO VARIANT LIMITING - Process all variants found
    if verbose and len(variants) > 50000:
        print(f"Processing all {len(variants)} variant observations, this could take a while...")
    
    # Check for positions in variants but not in coverage
    variant_positions = set(variants['pos'].unique())
    coverage_positions = set(coverage['pos'].unique())
    missing_positions = variant_positions - coverage_positions
    
    if missing_positions and verbose:
        print(f"Warning: {len(missing_positions)} positions in variants but not in coverage")
        if len(missing_positions) <= 10:
            print(f"Missing positions: {sorted(list(missing_positions))}")
        else:
            print(f"First 10 missing positions: {sorted(list(missing_positions))[:10]}")
    
    meta_cols = ['allele', 'pos', 'strand', 'ref']
    cell_cols = [col for col in variants.columns if col not in meta_cols]
    
    # MEMORY OPTIMIZATION: Process variants in chunks instead of all at once
    unique_positions = variants['pos'].unique()
    n_positions = len(unique_positions)
    
    
    # Split positions into chunks
    position_chunks = [unique_positions[i:i+chunk_size] for i in range(0, n_positions, chunk_size)]
    
    all_results = []
    
    for chunk_idx, pos_chunk in enumerate(position_chunks):
        # Get both variants and full merged data for this chunk
        chunk_variants = variants[variants['pos'].isin(pos_chunk)]
        merged_chunk = merged[merged['pos'].isin(pos_chunk)]
        
        # Pass both to process_variant_chunk
        chunk_result = process_variant_chunk(
            chunk_variants, merged_chunk, coverage, min_strand_count, cell_cols, coverage_threshold=coverage_threshold,
            verbose=(chunk_idx < 3)
        )
        
        if chunk_result is not None and len(chunk_result) > 0:
            all_results.append(chunk_result)
        
        # Force garbage collection after each chunk
        gc.collect()
    
    if not all_results:
        # Return empty result
        empty_df = pd.DataFrame(columns=['allele', 'pos', 'ref', 'variance', 
                                        'n_cells_conf_detected', 'n_cells_over_5', 'n_cells_over_10', 
                                        'n_cells_over_50', 'mean_coverage', 'strand_concordance', 'vaf_overall'] + cell_cols)
        empty_matrices = np.zeros((0, len(cell_cols)))
        return empty_df, empty_matrices, empty_matrices, None, None, empty_matrices
    
    
    final_df = pd.concat([r[0] for r in all_results], ignore_index=True)
    final_var_matrix = np.vstack([r[1] for r in all_results])
    final_vaf_matrix = np.vstack([r[2] for r in all_results])
    final_cov_matrix = np.vstack([r[5] for r in all_results])
    
    # For confident mask, we need to handle the DataFrame concatenation
    confident_dfs = [r[4] for r in all_results if r[4] is not None]
    if confident_dfs:
        final_confident_mask = pd.concat(confident_dfs, ignore_index=True)
    else:
        final_confident_mask = pd.DataFrame(np.zeros((len(final_df), len(cell_cols)), dtype=bool), columns=cell_cols)
    
    # Coverage matrix is not needed in chunked approach
    coverage_matrix_aligned = None
    
    
    return final_df, final_var_matrix, final_vaf_matrix, coverage_matrix_aligned, final_confident_mask, final_cov_matrix


def process_variant_chunk(chunk_variants, merged_chunk, coverage, min_strand_count, cell_cols, coverage_threshold=None, verbose=True):
    """
    Process a chunk of variants to avoid memory issues.
    """
    try:
        # Pivot to MultiIndex format for this chunk only
        pivoted = chunk_variants.set_index(['allele','pos', 'strand'])[cell_cols]
        pivoted = pivoted.astype(float)
        

        
        # Get forward and reverse matrices with error handling
        try:
            fwd_df = pivoted.xs('for', level='strand', drop_level=False)
            rev_df = pivoted.xs('rev', level='strand', drop_level=False)
        except KeyError as e:
            if verbose:
                print(f"  Skipping chunk due to missing strand data: {e}")
            return None
        
        # Align indices and apply filtering
        fwd_df = fwd_df.sort_index()
        rev_df = rev_df.sort_index()
        
        # Apply both-strands filtering (≥2 reads per strand)
        fwd_data = fwd_df.droplevel('strand')
        rev_data = rev_df.droplevel('strand')
        
        both_strands_mask = (fwd_data >= min_strand_count) & (rev_data >= min_strand_count)
        
        # Apply filtering
        fwd_filtered = fwd_data.where(both_strands_mask)
        rev_filtered = rev_data.where(both_strands_mask)
        
        # Sum forward + reverse for total counts
        grouped = (fwd_filtered.fillna(0) + rev_filtered.fillna(0)).reset_index()
        
        # Remove rows where all cell columns are 0
        grouped_nonan = grouped[grouped[cell_cols].sum(axis=1) > 0]
        
        if len(grouped_nonan) == 0:
            return None
        
        # Add back the 'ref' column that was lost during pivot - merge with original chunk_variants
        ref_mapping = chunk_variants[['allele', 'pos', 'ref']].drop_duplicates()
        grouped_nonan = grouped_nonan.merge(ref_mapping, on=['allele', 'pos'], how='left')
        ## NEW BIT 
        # Get ref allele counts for each variant position
        ref_counts = merged_chunk[
            (merged_chunk['allele'].str.upper() == merged_chunk['ref'].str.upper())
        ].groupby('pos')[cell_cols].sum()

        # Get total counts (ref + all non-ref) for each position
        total_counts = merged_chunk.groupby('pos')[cell_cols].sum()

        # Calculate VAF for each variant
        for idx, row in grouped_nonan.iterrows():
            pos = row['pos']
            # Sum across cells for this variant
            variant_sum = grouped_nonan.loc[idx, cell_cols].sum()
            # Get total sum for this position
            total_sum = total_counts.loc[pos].sum() if pos in total_counts.index else variant_sum
            # Store VAF
            grouped_nonan.loc[idx, 'vaf_overall'] = variant_sum / total_sum

# -------------------------------------------------------------------------------
        
        # Get coverage for these positions
        chunk_positions = set(grouped_nonan['pos'].unique())
        chunk_coverage = coverage[coverage['pos'].isin(chunk_positions)]
        
        # Build coverage matrix for this chunk only
        try:
            chunk_coverage_matrix = chunk_coverage.pivot(index='pos', columns='cell', values='coverage').fillna(0)
        except Exception as e:
            if verbose:
                print(f"  Warning: Coverage pivot failed for chunk: {e}")
            # Create dummy coverage matrix
            chunk_coverage_matrix = pd.DataFrame(0.0, 
                                               index=chunk_positions,
                                               columns=chunk_coverage['cell'].unique())
        
        # Extract coverage for each variant position
        variant_positions = grouped_nonan['pos'].tolist()
        
        # Replace the nested loop with this:
        cov_matrix = chunk_coverage_matrix.reindex(
            index=variant_positions, 
            columns=cell_cols, 
            fill_value=0.0
        ).values
        
        var_matrix = grouped_nonan[cell_cols].values
        
        # Compute VAF
        with np.errstate(divide='ignore', invalid='ignore'): 
            vaf_matrix = np.divide(var_matrix, cov_matrix, out=np.zeros_like(var_matrix, dtype=float), where=cov_matrix!=0)
            vaf_matrix[~np.isfinite(vaf_matrix)] = 0
        # Filter individual calls by coverage threshold (only if specified)
        if coverage_threshold is not None:
            vaf_matrix = np.where(cov_matrix >= coverage_threshold, vaf_matrix, 0)
        
        # Create confident mask for this chunk
        confident_mask_rows = []
        for idx, (_, row) in enumerate(grouped_nonan.iterrows()):
            allele = row['allele']
            pos = row['pos']
            
            # Find this variant in the original forward/reverse data
            fwd_mask = (fwd_data.index.get_level_values('allele') == allele) & (fwd_data.index.get_level_values('pos') == pos)
            rev_mask = (rev_data.index.get_level_values('allele') == allele) & (rev_data.index.get_level_values('pos') == pos)
            
            if fwd_mask.any() and rev_mask.any():
                fwd_values = fwd_data[fwd_mask].iloc[0]
                rev_values = rev_data[rev_mask].iloc[0]
                
                # Check both-strands requirement for each cell
                confident_cells = (fwd_values >= min_strand_count) & (rev_values >= min_strand_count)
                
                # Convert to list in same order as cell_cols
                confident_row = [confident_cells.get(cell, False) if cell in confident_cells.index else False for cell in cell_cols]
                confident_mask_rows.append(confident_row)
            else:
                # If no data found, mark all as non-confident
                confident_mask_rows.append([False] * len(cell_cols))
        
        confident_mask = pd.DataFrame(confident_mask_rows, columns=cell_cols)
        
        # Summary statistics for this chunk
        n_cells_conf_detected = np.sum(confident_mask.values, axis=1)
        n_cells_over_5 = np.sum((vaf_matrix >= 0.05) & confident_mask.values, axis=1)
        n_cells_over_10 = np.sum((vaf_matrix >= 0.10) & confident_mask.values, axis=1)
        n_cells_over_50 = np.sum((vaf_matrix >= 0.50) & confident_mask.values, axis=1)
        
        variance = np.nanvar(vaf_matrix, axis=1)
        mean_coverage = np.nanmean(cov_matrix, axis=1)
        
        # Calculate strand concordance efficiently using vectorized operations
        try:
            # Align forward and reverse data to same positions
            common_positions = fwd_data.index.intersection(rev_data.index)
            
            if len(common_positions) > 0:
                fwd_aligned = fwd_data.reindex(common_positions, fill_value=0)
                rev_aligned = rev_data.reindex(common_positions, fill_value=0)
                
                # Vectorized calculation: min/max ratio for each position
                with np.errstate(divide='ignore', invalid='ignore'):
                    min_counts = np.minimum(fwd_aligned.values, rev_aligned.values)
                    max_counts = np.maximum(fwd_aligned.values, rev_aligned.values)
                    
                    # Calculate concordance where both strands have data (max > 0)
                    concordance_matrix = np.divide(min_counts, max_counts, 
                                                 out=np.zeros_like(min_counts, dtype=float), 
                                                 where=max_counts > 0)
                    
                    # Average across cells for each variant (ignore zeros)
                    strand_concordance_aligned = np.array([
                        np.mean(row[row > 0]) if np.any(row > 0) else 0.0 
                        for row in concordance_matrix
                    ])
                
                # Map back to grouped_nonan order
                aligned_index_tuples = [(idx[0], idx[1]) for idx in common_positions]
                grouped_index_tuples = [(row['allele'], row['pos']) for _, row in grouped_nonan.iterrows()]
                
                strand_concordance = []
                for gt in grouped_index_tuples:
                    if gt in aligned_index_tuples:
                        idx = aligned_index_tuples.index(gt)
                        strand_concordance.append(strand_concordance_aligned[idx])
                    else:
                        strand_concordance.append(0.0)
                
                strand_concordance = np.array(strand_concordance)
            else:
                # No common positions
                strand_concordance = np.zeros(len(grouped_nonan))
        except Exception as e:
            # Fallback to default values if calculation fails
            strand_concordance = np.full(len(grouped_nonan), 0.5)
        
        # Add summary columns
        grouped_nonan['variance'] = variance
        grouped_nonan['n_cells_conf_detected'] = n_cells_conf_detected
        grouped_nonan['n_cells_over_5'] = n_cells_over_5
        grouped_nonan['n_cells_over_10'] = n_cells_over_10
        grouped_nonan['n_cells_over_50'] = n_cells_over_50
        grouped_nonan['mean_coverage'] = mean_coverage
        grouped_nonan['strand_concordance'] = strand_concordance
        
           
        
        
        return (grouped_nonan, var_matrix, vaf_matrix, None, confident_mask, cov_matrix)
        
    except Exception as e:
        if verbose:
            print(f"  Error processing chunk: {e}")
        return None
def process_variants_fast(input_folder: str, 
                         adata: anndata.AnnData,
                         min_strand_count: int = 2,
                         coverage_threshold: int = None,
                         verbose: bool = True,
                         chunk_size: int = 1000) -> anndata.AnnData:
  
    """
    Fast variant calling pipeline - based on original with bug fixes.
    """
    if verbose:
        print("=== Fast Variant Calling Pipeline ===\n\n")
        print("You'll find variant names in adata.uns['variant_names']\n")
        print("Find coverage PER cell in adata.obsm['coverage_per_cell']\n")
        print("Variant summmary stats similar to Signac are in adata.uns['variant_summary']\n")
        print('Variant matrices are in adata.obsm["variant_vaf"]\n\n')
        print("=== IMPORTANT NOTE ===\n")
        print("Remember that variant VAFs are only included if they pass the min_strand_count filter\n")
        print("AND the coverage threshold if specified!\n")
        print("======================================\n\n")

    
    try:
        counts_df = combine_allele_counts(input_folder)
        if verbose:
            print(f"Loaded counts data: {counts_df.shape}")
    except Exception as e:
        if verbose:
            print(f"Error loading allele counts: {e}")
        raise
    
    
    try:
        refallele_df = read_refallele(input_folder)

    except Exception as e:
        if verbose:
            print(f"Error loading refallele: {e}")
        raise

    
    # Identify variant positions from counts data
    # CRITICAL: Ensure both DataFrames have same data type for 'pos' column
    counts_df['pos'] = counts_df['pos'].astype(str)
    refallele_df['pos'] = refallele_df['pos'].astype(str)
    
    merged_preview = counts_df.merge(refallele_df, on='pos', how='left')
    # Ensure we don't treat missing refs as variants
    merged_preview = merged_preview.dropna(subset=['ref'])
    variant_positions = set(merged_preview[merged_preview['allele'].str.upper() != merged_preview['ref'].str.upper()]['pos'].unique())
    # Exclude known fake variant positions prior to coverage read for consistency
    fake_variants = ['3107', '301', '302', '309', '310', '311', '316', '3108', '3109', '16182', '16183']
    variant_positions = {p for p in variant_positions if p not in fake_variants}
    
 
        
    try:
        coverage = read_coverage(input_folder, variant_positions=variant_positions)
        if verbose:
            
            print(f"Loaded coverage data: {coverage.shape}")
    except Exception as e:
        if verbose:
            print(f"Error loading coverage: {e}")
        raise
    
    # Step 2: Run variant calling (fast method)
    summary_df, var_matrix, vaf_matrix, coverage_matrix_aligned, confident_mask, actual_coverage_matrix = identify_variants_fast(
        counts_df=counts_df,
        refallele_df=refallele_df,
        coverage=coverage,
        min_strand_count=min_strand_count,
        coverage_threshold=coverage_threshold,
        verbose=verbose
    )
    
    # Step 3: Quick integration with AnnData
    if verbose:
        print("Integrating with AnnData...")
    
    # Find matching cells
    cell_barcodes = adata.obs.index.astype(str)
    cell_cols = list(summary_df.columns.difference([
        "allele", "pos", "ref",
        "variance", "n_cells_conf_detected",
        "n_cells_over_5", "n_cells_over_10",
        "n_cells_over_50", "mean_coverage",
        "strand_concordance", "vaf_overall"
    ]))
    matching_cells = [c for c in cell_barcodes if c in cell_cols]
    
    if not matching_cells:
        raise ValueError("No matching cell barcodes")
    
    # Subset matrices to matching cells
    cell_indices = [cell_cols.index(cb) for cb in matching_cells]
    vaf_subset = vaf_matrix[:, cell_indices]
    var_subset = var_matrix[:, cell_indices]
    confident_subset = confident_mask.iloc[:, cell_indices]
    
    # Extract coverage matrix for the same variants and cells using the SAME coverage used in VAF calculation
    variant_positions = summary_df['pos'].tolist()
    
    # Use the EXACT coverage matrix that was used for VAF calculation
    # actual_coverage_matrix is already aligned to the final variants and all cells
    cov_subset = actual_coverage_matrix[:, cell_indices]
    
    # Store in AnnData efficiently, fix to make variant names the correct way round 
    variant_names = (
    summary_df[["ref", "pos", "allele"]]
    .astype(str)
    .apply(lambda col: col.str.upper())
    .agg("-".join, axis=1)
    .tolist()
    )


    adata.uns['variant_summary'] = summary_df[["variance", "n_cells_conf_detected", "n_cells_over_5", "n_cells_over_10", "n_cells_over_50", "mean_coverage", "strand_concordance", "vaf_overall"]].copy()
    adata.uns['variant_names'] = variant_names
    
    # Create properly shaped matrices for AnnData (cells x variants)
    vaf_matrix_cells_variants = vaf_subset.T  # Should be (cells, variants)
    var_matrix_cells_variants = var_subset.T  # Should be (cells, variants)
    cov_matrix_cells_variants = cov_subset.T  # Should be (cells, variants)
    confident_matrix_cells_variants = confident_subset.T.values  # Should be (cells, variants)

    
    # Reindex to match adata cell order
    vaf_full = pd.DataFrame(vaf_matrix_cells_variants, index=matching_cells, columns=variant_names).reindex(adata.obs.index, fill_value=0)
    var_full = pd.DataFrame(var_matrix_cells_variants, index=matching_cells, columns=variant_names).reindex(adata.obs.index, fill_value=0)
    cov_full = pd.DataFrame(cov_matrix_cells_variants, index=matching_cells, columns=variant_names).reindex(adata.obs.index, fill_value=0)
    conf_full = pd.DataFrame(confident_matrix_cells_variants, index=matching_cells, columns=variant_names).reindex(adata.obs.index, fill_value=False)
    
    # Store matrices in obsm (these should be 2D: cells x variants)
    adata.obsm['variant_vaf'] = vaf_full.values
    adata.obsm['variant_counts'] = var_full.values
    adata.obsm['coverage_per_cell'] = cov_full.values  #Coverage per variant per cell
    adata.obsm['variant_confident'] = conf_full.values

    
    # Add summary stats
    adata.obs['total_variant_count'] = np.sum(adata.obsm['variant_counts'], axis=1)
    
    
    if verbose:
        print("=== Pipeline Complete ===")
        print(f"Processed {len(variant_names)} variants in {len(matching_cells)} matching cells")
    
    return adata




if __name__ == "__main__":
    print("Fast Variant Calling Pipeline\n")
    