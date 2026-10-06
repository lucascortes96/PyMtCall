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
    exactly one of the alleles (A, C, G or T), suggesting potential false negatives
    due to insufficient coverage on the other allele(s).

    Parameters:
            file_path (str): Directory containing ``mgatk.A.txt.gz``, ``mgatk.C.txt.gz``,
                    ``mgatk.G.txt.gz`` and ``mgatk.T.txt.gz``.
            position (int): mtDNA coordinate to evaluate (e.g., ``3243``).
            min_reads (int): Minimum reads required on both forward and reverse
                    strands for a cell to be considered adequately covered.

    Input files format:
            - ``mgatk.A.txt.gz``, ``mgatk.C.txt.gz``, ``mgatk.G.txt.gz``, ``mgatk.T.txt.gz``:
              gzip-compressed CSV with columns ``pos, cell, forw, rev``
            - ``pos``: mtDNA position
            - ``cell``: cell barcode
            - ``forw``/``rev``: read counts on forward and reverse strands

    Returns:
            pandas.DataFrame: Single-column DataFrame (``cell``) listing barcodes
            present in exactly one adequately covered allele set.

    Notes:
            - Coverage criterion: ``forw >= min_reads`` AND ``rev >= min_reads``.
            - The result uses the symmetric difference between the adequately
                covered allele cell sets:
                    * Cells present in more than one set are not flagged.
                    * Cells present in exactly one set are candidates for false negatives.
    """

    def allele_file(base):
        for suffix in (".txt.gz", ".txt"):
            candidate = path.join(file_path, f"mgatk.{base}{suffix}")
            if path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No mgatk.{base}.txt[.gz] found in {file_path}")

    try:
        A_path = allele_file("A")
        C_path = allele_file("C")
        G_path = allele_file("G")
        T_path = allele_file("T")

        df_A = pd.read_csv(A_path, sep=",", header=None,
                           names=["pos", "cell", "forw", "rev"],
                           on_bad_lines="skip")

        df_C = pd.read_csv(C_path, sep=",", header=None,
                           names=["pos", "cell", "forw", "rev"],
                           on_bad_lines="skip")

        df_G = pd.read_csv(G_path, sep=",", header=None,
                           names=["pos", "cell", "forw", "rev"],
                           on_bad_lines="skip")

        df_T = pd.read_csv(T_path, sep=",", header=None,
                           names=["pos", "cell", "forw", "rev"],
                           on_bad_lines="skip")

        pos_A = df_A[(df_A["pos"] == position) & (df_A["forw"] >= min_reads) & (df_A["rev"] >= min_reads)]
        pos_C = df_C[(df_C["pos"] == position) & (df_C["forw"] >= min_reads) & (df_C["rev"] >= min_reads)]
        pos_G = df_G[(df_G["pos"] == position) & (df_G["forw"] >= min_reads) & (df_G["rev"] >= min_reads)]
        pos_T = df_T[(df_T["pos"] == position) & (df_T["forw"] >= min_reads) & (df_T["rev"] >= min_reads)]

        unique_cells = (
            set(pos_A["cell"])
            ^ set(pos_C["cell"])
            ^ set(pos_G["cell"])
            ^ set(pos_T["cell"])
        )

        unique_cells_df = pd.DataFrame(unique_cells, columns=["cell"])
        return unique_cells_df

    except Exception as e:
        raise FileNotFoundError(
            f"Error accessing files in {file_path}: {e}, is your file gzipped?"
        )

