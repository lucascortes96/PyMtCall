# PyMtCall

PyMtCall provides tools for single-cell ATAC-seq and multiomic mitochondrial variant calling,
clonotype analysis, and false-negative checks using mgatk-style outputs.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/lucascortes96/PyMtCall
```

Then import with:

```python
import Py_mTCall_tools
```

## Current Public API

The package currently exports the following functions:

### Variant calling (`variant_calling_fast`)
- `read_allele_file(path, allele)`
- `build_sparse_matrix(df, cell_lookup, strand)`
- `combine_allele_counts(directory=None)`
- `read_refallele(directory)`
- `read_coverage(directory, variant_positions=None)`
- `process_variants_fast(input_folder, adata, min_strand_count=2, coverage_threshold=None, verbose=True, chunk_size=1000)`

### Clonotype analysis (`clonotype_analysis`)
- `cluster_clonotypes(adata, layer=None, group_by=None)`
- `find_clonotypes(adata, layer=None, features=None, metric='cosine', resolution=1.0, k=10, algorithm='leiden')`
- `analyze_clonotypes_from_variants(adata, vaf_layer_name='vaf', min_cells=1, min_vaf=0.1, min_coverage=20, min_strand_concordance=0.3, max_variance=1, resolution=1.0, k=10)`
- `plot_clonotype_vaf_heatmap(adata, vaf_layer_name='vaf', clonotype_col='clonotype_leiden', top_variants=50, figsize=(12, 10), save=None)`
- `set_if_null(x, y)`

### False negative checking (`FalseNegativeChecker`)
- `check_false_negatives(file_path, position, min_reads)`
- `use for single variants at a time, will give slightly more accurate VAF data but is resource intensive`

## Usage

### 1) Fast variant calling

```python
import scanpy as sc
from Py_mTCall_tools import process_variants_fast

adata = sc.read_h5ad("your_data.h5ad")
adata = process_variants_fast(
    input_folder="/path/to/mgatk/final",
    adata=adata,
    min_strand_count=2,
    coverage_threshold=None,
    verbose=True
)
```

Expected outputs added to `adata`:
- `adata.uns['variant_summary']`
- `adata.uns['variant_names']`
- `adata.obsm['variant_vaf']`
- `adata.obsm['variant_counts']`
- `adata.obsm['coverage_per_cell']`
- `adata.obsm['variant_confident']`

### 2) Clonotype analysis from called variants

```python
from Py_mTCall_tools import analyze_clonotypes_from_variants, plot_clonotype_vaf_heatmap

adata = analyze_clonotypes_from_variants(
    adata,
    min_cells=5,
    min_vaf=0.01,
    min_coverage=20,
    resolution=1.0,
    k=10
)

fig = plot_clonotype_vaf_heatmap(
    adata,
    clonotype_col="clonotype_leiden",
    top_variants=30
)
```

### 3) False negative check at a mitochondrial position

```python
from Py_mTCall_tools import check_false_negatives

false_neg_cells = check_false_negatives(
    file_path="/path/to/mgatk/final",
    position=3243,
    min_reads=2
)
```

`check_false_negatives` returns a DataFrame of cells that have adequate
bidirectional coverage in exactly one allele set. Cells that do not have sufficient coverage
are flagged as potential false negatives. 
## Requirements

- numpy
- pandas
- anndata
- scanpy
- scipy
- scikit-learn
- matplotlib
- seaborn

Optional:
- python-igraph (used by leiden/louvain workflows; fallback behavior exists)

## Examples

See the `examples/` directory for end-to-end notebook and script usage.

## License

MIT