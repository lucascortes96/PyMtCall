#!/usr/bin/env python3
'''
Comprehensively testing variant calling using the generated test data found in this
directory. Test data is comprised of 5 cells with generated coverage for each allele. 
Coverage is for each position in each cell is between 0-25
'''
import sys
sys.path.insert(0, '/Users/lucascortes/Library/CloudStorage/OneDrive-Personal/Documents/Work/Pickett/ATAC-Seq/Py-mTCall')
from Py_mTCall_tools import variant_calling as normal 
from Py_mTCall_tools import variant_calling_fast as fast
from Py_mTCall_tools import variant_calling_ultra_fast as ultra
import pandas as pd
import time 
import cProfile
import psutil
import os

## Reading in the reference allele and position
start = time.time()
ref_dataframe = fast.read_refallele(".")
print(ref_dataframe)
print(f"Time to read refallele: {time.time() - start:.2f}s")
## Reading in coverage and position
start = time.time()
coverage_dataframe = fast.read_coverage(".")
print(coverage_dataframe.head()) 
print(f"Time to read coverage: {time.time() - start:.2f}s")
## Combining allele counts with strand information at each position in each cell
start = time.time()
combined_df = fast.combine_allele_counts(".")
print(combined_df.head(50))
print(f"Time to combine alleles: {time.time() - start:.2f}s")
## This does the bulk of the processing, does VAF calculations, strand calculations etc
start = time.time()
process = psutil.Process(os.getpid())
identified_variants = fast.identify_variants_fast(combined_df, ref_dataframe, coverage_dataframe)
print(f"Time to identify variants: {time.time() - start:.2f}s")
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
# Refer final table back to final_vaf to verify that VAF is calculated correctly
final_table = identified_variants[0]
final_table.to_csv("csv_outputs/final_variant_table.csv")
final_vaf = identified_variants[2]
print(final_vaf)


