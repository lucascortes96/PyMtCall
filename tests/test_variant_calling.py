# This file can be used for basic tests or examples of usage for the snapatac_tools package.
# You can expand this with pytest or unittest for more robust testing.

import anndata
from snapatac_tools import process_and_integrate_variants

def test_process_and_integrate_variants():
    # Example usage (requires test data in 'test_data' folder and a test AnnData object)
    adata = anndata.AnnData(X=[[0]])  # Replace with real test AnnData
    adata_new = process_and_integrate_variants('test_data', adata)
    print(adata_new)

if __name__ == "__main__":
    test_process_and_integrate_variants()