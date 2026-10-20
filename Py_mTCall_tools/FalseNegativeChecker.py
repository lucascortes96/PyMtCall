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
def check_false_negatives(file_path, position, min_reads, base):
    
    """Identify cells with adequate bidirectional coverage for exactly one
    of the three alleles other than the specified base at a given mtDNA
    position, helping characterise potential false-negative candidates.

    Parameters:
        file_path (str): Directory containing ``mgatk.A.txt.gz``,
            ``mgatk.C.txt.gz``, ``mgatk.G.txt.gz`` and ``mgatk.T.txt.gz``.
        position (int): mtDNA coordinate to evaluate (e.g., ``3243``).
        min_reads (int): Minimum reads required on both forward and reverse
            strands for a cell to be considered adequately covered.
        base (str): Allele to exclude from the analysis, typically the mutant
            allele (A, C, G or T).

    Input files format:
        - ``mgatk.A.txt.gz``, ``mgatk.C.txt.gz``, ``mgatk.G.txt.gz``,
        ``mgatk.T.txt.gz`` (or uncompressed ``.txt`` equivalents).
        - Each file contains columns ``pos, cell, forw, rev``.

    Returns:
        pandas.DataFrame: Single-column DataFrame (``cell``) listing barcodes
        with adequate coverage for exactly one of the three alleles other
        than ``base`` at the specified position.

    Notes:
        - Adequate coverage requires ``forw >= min_reads`` AND
        ``rev >= min_reads``.
        - The specified ``base`` is excluded from the coverage comparison.
        - Cells are flagged if they appear in exactly one of the three
        remaining allele sets; cells appearing in zero, two or three
        sets are not returned.
        - The output characterises alternative-allele coverage in cells
        assessed separately for coverage of the excluded allele. It does
        not independently confirm a cell's genotype or determine whether
        a false-negative variant call occurred.
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

        base = base.upper()
        if base not in {"A", "C", "G", "T"}:
            raise ValueError("base must be one of A, C, G, or T")

        allele_dfs = {
            "A": df_A,
            "C": df_C,
            "G": df_G,
            "T": df_T,
        }

        covered_cells = {}

        for allele, df in allele_dfs.items():
            if allele == base:
                continue

            filtered = df[
                (df["pos"] == position)
                & (df["forw"] >= min_reads)
                & (df["rev"] >= min_reads)
            ]
            covered_cells[allele] = set(filtered["cell"])

        # Count how many non-mutant allele sets each cell appears in
        from collections import Counter

        counts = Counter(
            cell
            for cells in covered_cells.values()
            for cell in cells
        )

        # Keep cells appearing in exactly one adequately covered allele set
        unique_cells = [
            cell for cell, count in counts.items()
            if count == 1
        ]
        return pd.DataFrame(unique_cells, columns=["cell"])

    except Exception as e:
        raise FileNotFoundError(
            f"Error accessing files in {file_path}: {e}, is your file gzipped?"
        )

