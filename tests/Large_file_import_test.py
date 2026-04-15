'''
Testing efficiency of reading in allele files and then flipping the dataframe in pandas.
This is due to observed poor performance of reading in files when they are quite big
and data rich. Not a problem when files are smaller or full of zeros. 
'''

import pandas as pd
from scipy.sparse import coo_matrix
import os
import glob
import numpy as np

def read_allele_file(path, allele):
    """Read a single allele count file and annotate with allele and strand information."""
    df = pd.read_csv(path, sep=",", header=None, names=["pos", "cell", "forw", "rev"])
    print(df['pos'].memory_usage())
    df["pos"] = df["pos"].astype(str)
    print(df['pos'].memory_usage())
    df["row_forw"] = f"{allele}-" + df["pos"] + "-for"
    print(df['row_forw'].memory_usage())
    df["row_rev"] = f"{allele}-" + df["pos"] + "-rev"
    print(df.head())
    return df

#read_allele_file("/Users/lucascortes/Documents/Pickett/mtscATAC-Seq/2025_049_Sarah_Pickett/MC3_test/final/mgatk.A.txt.gz", "A")
'''
More memory efficient version
Pandas:

Identifies all unique string values.

Stores them once in a small list (.cat.categories).

Replaces your column with integer codes pointing into that list.

So instead of storing "13-for-A" 10,000 times, it stores it once, and references it with small integers.

'''

def read_allele_file2(path, allele):
    df = pd.read_csv(path, sep=",", header=None, names=["pos", "cell", "forw", "rev"])
    df['row_forw'] = (allele + '-' + df['pos'].astype(str) + '-for').astype('category')
    df['row_rev'] = (allele + '-' + df['pos'].astype(str) + '-rev').astype('category')
    print(df['row_forw'].memory_usage())
    print(df.head())
    return df

#read_allele_file2("/Users/lucascortes/Documents/Pickett/mtscATAC-Seq/2025_049_Sarah_Pickett/MC3_test/final/mgatk.A.txt.gz", "A")

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
    print("Starting")
    alleles = ["A", "C", "G", "T"]
    all_dfs = []
    
    for allele in alleles:
        # Look for both compressed and uncompressed files
        pattern_gz = os.path.join(directory if directory else os.getcwd(), f"*.{allele}.txt.gz")
        pattern_txt = os.path.join(directory if directory else os.getcwd(), f"*.{allele}.txt")
        files = glob.glob(pattern_gz) + glob.glob(pattern_txt)
        for path in files:
            print(f"Reading allele file: {path}")
            all_dfs.append(read_allele_file2(path, allele))
    
    all_df = pd.concat(all_dfs, ignore_index=True)
    print(all_df.head())
    
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
    print(multi_index)
    
    # Build DataFrame and IMMEDIATELY filter out 0-count rows (BUG FIX)
    df = pd.DataFrame.sparse.from_spmatrix(all_mat, index=multi_index, columns=cellbarcodes)
    
    # Convert to dense for filtering (this is the key fix)
    df_dense = df.sparse.to_dense()
    non_zero_rows = (df_dense != 0).any(axis=1)
    df_filtered = df_dense[non_zero_rows]
    
    # Convert back to the expected format
    df_filtered = df_filtered.reset_index()  # This makes 'allele', 'pos', 'strand' columns
    return df_filtered


combine_allele_counts("/Users/lucascortes/Documents/Pickett/mtscATAC-Seq/2025_049_Sarah_Pickett/MC3_test/final")

