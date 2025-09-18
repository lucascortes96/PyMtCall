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

from .clonotype_analysis import (
    cluster_clonotypes,
    find_clonotypes,
    plot_clonotype_heatmap,
    analyze_clonotypes_from_variants,
    plot_clonotype_vaf_heatmap,
    get_variant_confidence_summary,
    set_if_null,
)
