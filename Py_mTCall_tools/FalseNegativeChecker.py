"""Utilities for assessing potential false negatives at a mitochondrial SNV site
(e.g., m.3243A>G) using per-allele coverage files produced by mgatk.

A true negative requires sufficient bidirectional coverage (>= ``min_reads`` on
both forward and reverse strands) to confidently call the absence of the
alternate allele. Cells lacking sufficient coverage on one allele may be
flagged as potential false negatives.
"""

import pandas as pd
from os import path


##Checking false negative rate for ref allele
def check_false_negatives(file_path, position, min_reads):
    """Identify cells with adequate coverage at a given mtDNA position for
    exactly one of the alleles (A or G), suggesting potential false negatives
    due to insufficient coverage on the other allele.

    Parameters:
            file_path (str): Directory containing ``mgatk.A.txt.gz`` and
                    ``mgatk.G.txt.gz``.
            position (int): mtDNA coordinate to evaluate (e.g., ``3243``).
            min_reads (int): Minimum reads required on both forward and reverse
                    strands for a cell to be considered adequately covered.

    Input files format:
            - ``mgatk.A.txt.gz`` and ``mgatk.G.txt.gz``: gzip-compressed CSV with
                columns ``pos, cell, forw, rev``
            - ``pos``: mtDNA position
            - ``cell``: cell barcode
            - ``forw``/``rev``: read counts on forward and reverse strands

    Returns:
            pandas.DataFrame: Single-column DataFrame (``cell``) listing barcodes
            present in exactly one adequately covered allele set.

    Notes:
            - Coverage criterion: ``forw >= min_reads`` AND ``rev >= min_reads``.
            - The result uses the symmetric difference between the adequately
                covered A-allele and G-allele cell sets:
                    * Cells present in both sets have good coverage for both alleles
                        and are not flagged.
                    * Cells present in neither set lack coverage altogether and are not
                        included.
                    * Cells present in exactly one set are candidates for false
                        negatives because one allele lacks sufficient coverage.

    Example:
            >>> check_false_negatives("/path/to/mgatk_out/final", 3243, 2)
    """
    try:
        A_path = path.join(file_path, "mgatk.A.txt.gz")
        G_path = path.join(file_path, "mgatk.G.txt.gz")
        df_A = pd.read_csv(A_path, sep=",", header=None, names=["pos", "cell", "forw", "rev"], on_bad_lines="skip" )

        pos_3243_A = df_A[(df_A["pos"] == position) & (df_A["forw"] >= min_reads) & (df_A["rev"] >= min_reads)]

        ##Checking false negative rate for alt allele
        df_G = pd.read_csv(G_path, sep=",", header=None, names=["pos", "cell", "forw", "rev"], on_bad_lines="skip" )

        pos_3243_G = df_G[(df_G["pos"] == position) & (df_G["forw"] >= min_reads) & (df_G["rev"] >= min_reads)]

        ## Gets the number of unique cells in either dataframe
        unique_cells = set(pos_3243_A["cell"]) ^ (set(pos_3243_G["cell"]))
        ## Converted to dataframe so I can pull into other analysis 
        unique_cells_df = pd.DataFrame(unique_cells, columns=["cell"])
        return unique_cells_df
    except Exception as e:
        raise FileNotFoundError(f"Error accessing files in {file_path}: {e}, is your file gzipped?")

    

'''
#### Example usage ####:
print(check_false_negatives("/Users/lucascortes/Documents/Pickett/mtscATAC-Seq/2026_003_Sarah_Pickett/outs/WBC063_outs/mgatk_out/final", 3243, 2))
'''

