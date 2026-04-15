#!/usr/bin/env python3
"""
Generate random test data for variant calling pipeline.
Randomly distributes coverage reads across 4 allele files (A, T, C, G) with 2 strands each.
"""

import pandas as pd
import numpy as np
import os

def generate_random_allele_data(coverage_file, output_dir):
    """
    Generate random allele count files from coverage data.
    
    Args:
        coverage_file: Path to coverage file (pos,cell,coverage)
        output_dir: Directory to write allele files
    """
    
    # Read coverage data
    print(f"Reading coverage data from {coverage_file}")
    coverage_df = pd.read_csv(coverage_file, header=None, names=['pos', 'cell', 'coverage'])
    print(f"Loaded {len(coverage_df)} coverage entries")
    
    # Initialize allele data storage
    alleles = ['A', 'T', 'C', 'G']
    allele_data = {allele: [] for allele in alleles}
    
    # Process each coverage entry
    for idx, row in coverage_df.iterrows():
        pos = row['pos']
        cell = row['cell']
        total_coverage = row['coverage']
        
        if total_coverage <= 0:
            continue
            
        # Randomly distribute total_coverage reads across 8 categories:
        # 4 alleles × 2 strands each = 8 total categories
        
        # Use multinomial distribution for random assignment
        # Each of the 8 categories gets equal probability
        n_categories = 8  # 4 alleles × 2 strands
        probabilities = np.ones(n_categories) / n_categories  # Equal probability
        
        # Generate random distribution
        random_counts = np.random.multinomial(total_coverage, probabilities)
        
        # Assign counts to alleles and strands
        count_idx = 0
        for allele in alleles:
            forward_count = random_counts[count_idx]
            reverse_count = random_counts[count_idx + 1]
            count_idx += 2
            
            # Add ALL entries, even if both strands are 0
            allele_data[allele].append({
                'pos': pos,
                'cell': cell,
                'forward': forward_count,
                'reverse': reverse_count
            })
    
    # Write allele files
    os.makedirs(output_dir, exist_ok=True)
    
    for allele in alleles:
        output_file = os.path.join(output_dir, f"test.{allele}.txt")
        
        # Convert to DataFrame and write (now all alleles will have data)
        df = pd.DataFrame(allele_data[allele])
        df = df[['pos', 'cell', 'forward', 'reverse']]  # Ensure column order
        df.to_csv(output_file, index=False, header=False)
        print(f"Wrote {len(df)} entries to {output_file}")
    
    # Print summary statistics
    print("\nSummary:")
    total_original_coverage = coverage_df['coverage'].sum()
    total_distributed_reads = 0
    
    for allele in alleles:
        allele_total = sum(entry['forward'] + entry['reverse'] for entry in allele_data[allele])
        total_distributed_reads += allele_total
        non_zero_entries = sum(1 for entry in allele_data[allele] if entry['forward'] > 0 or entry['reverse'] > 0)
        zero_entries = len(allele_data[allele]) - non_zero_entries
        print(f"  {allele}: {allele_total} reads across {len(allele_data[allele])} entries ({non_zero_entries} with reads, {zero_entries} with 0,0)")
    
    print(f"\nTotal coverage from input: {total_original_coverage}")
    print(f"Total reads distributed: {total_distributed_reads}")
    print(f"Conservation check: {'✓ PASSED' if total_original_coverage == total_distributed_reads else '✗ FAILED'}")


if __name__ == "__main__":
    # Set random seed for reproducibility (remove this line for true randomness)
    np.random.seed(42)
    
    # File paths
    coverage_file = "test.coverage.txt"
    output_dir = "."  # Current directory
    
    print("Generating random test data for variant calling...")
    print(f"Input: {coverage_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    generate_random_allele_data(coverage_file, output_dir)
    
    print("=" * 50)
    print("Test data generation complete!")