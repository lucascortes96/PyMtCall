# snapatac_tools/__init__.py

from .variant_calling import (
    read_allele_file,
    build_sparse_matrix,
    find_allele_files,
    combine_allele_counts,
    read_refallele,
    read_coverage,
    identify_variants,
    process_and_integrate_variants,
)

__version__ = "0.1.0"