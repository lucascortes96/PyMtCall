# PyMtCall

This package provides functions for efficient single-cell ATAC-seq mitchondrial variant calling and integration with Scanpy/AnnData workflows.

## Installation

Clone the repository and install with pip:

```bash
pip install git+https://github.com/lucascortes96/PyMtCall
```

## Usage Example

```python
import scanpy as sc
from snapatac_tools import process_and_integrate_variants

adata = sc.read_h5ad('your_data.h5ad')
adata_new = process_and_integrate_variants('input_folder', adata)
```

- `input_folder` should contain your allele count, refAllele, and coverage files.
- The resulting `adata_new` will have variant and VAF information integrated for downstream analysis.

## Requirements
- numpy
- pandas
- anndata
- scanpy
- scipy

## License
MIT