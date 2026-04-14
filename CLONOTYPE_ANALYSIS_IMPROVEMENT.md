# Clonotype Analysis Improvement: Confident Variant Filtering

## Problem Identified
The original clonotype analysis was using ALL detected variants, including low-quality and potentially spurious variants. This led to the creation of artificial clonotypes based on noisy data rather than genuine mutational patterns.

## Solution Implemented
Updated the `analyze_clonotypes_from_variants()` function to use only **confidently detected variants** with multiple quality criteria:

### New Confidence Filters Applied:

1. **Minimum Confident Detection** (`min_cells=10`)
   - Variants must be confidently detected in ≥10 cells
   - Uses `n_cells_conf_detected` metric from variant calling

2. **High VAF Threshold** (`min_vaf=0.1`) 
   - Variants must have VAF ≥10% in ≥10 cells
   - Uses `n_cells_over_10` metric from variant calling

3. **Coverage Depth** (`min_coverage=20`)
   - Variants must have mean coverage ≥20 reads
   - Ensures sufficient sequencing depth for reliable detection

4. **Strand Concordance** (`min_strand_concordance=0.3`)
   - Forward and reverse strand counts must be correlated (r ≥ 0.3)
   - Filters out strand-biased artifacts
   - Accepts NaN values (single-strand variants)

5. **Low Variance** (`max_variance=0.1`)
   - VAF variance across cells must be ≤0.1
   - Filters out noisy variants with inconsistent detection

## Key Benefits:

- **Eliminates spurious clonotypes** caused by low-quality variants
- **Improves clonotype accuracy** by using only reliable mutational signals
- **Provides detailed filtering feedback** showing how many variants pass each criterion
- **Maintains flexibility** with adjustable confidence thresholds

## Usage Example:

```python
# Import the improved function
from Py_mTCall_tools import analyze_clonotypes_from_variants, get_variant_confidence_summary

# Check variant quality first
confidence_summary = get_variant_confidence_summary(adata)
print(confidence_summary)

# Run confident clonotype analysis
adata_clonotypes = analyze_clonotypes_from_variants(
    adata,
    min_cells=10,              # Confident detection in ≥10 cells
    min_vaf=0.1,              # VAF ≥10% threshold
    min_coverage=20,          # Mean coverage ≥20x
    min_strand_concordance=0.3, # Strand correlation ≥0.3
    max_variance=0.1          # VAF variance ≤0.1
)
```

## Updated Files:
- `Py_mTCall_tools/clonotype_analysis.py`: Enhanced variant filtering logic
- `Py_mTCall_tools/__init__.py`: Added new confidence summary function export
- `examples/PymTCallExample.ipynb`: Updated with confidence assessment and improved parameters

This improvement ensures that clonotype analysis reflects genuine biological variation rather than technical artifacts.
