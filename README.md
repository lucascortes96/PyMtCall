# PyMtCall

This package provides functions for efficient single-cell ATAC-seq mitchondrial variant calling, clonotype analysis, and integration with Scanpy/AnnData workflows.

## Installation

Clone the repository and install with pip:

```bash
pip install git+https://github.com/lucascortes96/PyMtCall
```

## Features

- **Variant Calling**: Process MGATK output files to identify mitochondrial variants
- **Clonotype Analysis**: Identify groups of related cells based on variant allele frequencies
- **Scanpy Integration**: Seamless integration with single-cell analysis workflows
- **Visualization**: Built-in plotting functions for variant and clonotype analysis

## Usage Example

### Basic Variant Calling

```python
import scanpy as sc
from Py_mTCall_tools import process_and_integrate_variants

adata = sc.read_h5ad('your_data.h5ad')
adata = process_and_integrate_variants('input_folder', adata)
```

### Clonotype Analysis

```python
from Py_mTCall_tools import analyze_clonotypes_from_variants, plot_clonotype_vaf_heatmap

# Analyze clonotypes based on variant patterns
adata = analyze_clonotypes_from_variants(
    adata,
    min_cells=5,
    min_vaf=0.05,
    resolution=1.0
)

# Visualize clonotype VAF patterns
fig = plot_clonotype_vaf_heatmap(adata, top_variants=50)
```

- `input_folder` should contain your MGATK output; allele count, refAllele, and coverage files.
- The resulting `adata` will have variant, VAF, and clonotype information integrated for downstream analysis.

## Documentation

For detailed documentation on clonotype analysis, see [CLONOTYPE_ANALYSIS.md](CLONOTYPE_ANALYSIS.md).

## Requirements
- numpy
- pandas
- anndata
- scanpy
- scipy
- scikit-learn
- matplotlib
- seaborn

### Optional
- python-igraph (for advanced clustering algorithms)
- scipy

## Examples 
- Please go to examples/ to find usage 

## License
MIT