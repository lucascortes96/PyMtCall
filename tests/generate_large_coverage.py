#!/usr/bin/env python3
"""
Generate large test coverage file with 1000 cells
"""
import numpy as np
import pandas as pd

def generate_large_coverage():
    """Generate coverage file with 1000 cells, positions 1-10, coverage 1-25"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    coverage_data = []

    # Generate for 1000 cells
    for cell_num in range(1, 1001):  # cell1 to cell1000
        cell_name = f"cell{cell_num}"

        # For each position (1-10000)
        for pos in range(1, 10001):  # Positions 1 to 10000
            # Random coverage between 1 and 25
            coverage = np.random.randint(1, 26)  # 1 to 25 inclusive
            
            coverage_data.append([pos, cell_name, coverage])
    
    # Create DataFrame and save
    df = pd.DataFrame(coverage_data, columns=['pos', 'cell', 'coverage'])
    df.to_csv('test.coverage.txt', header=False, index=False)
    
    print(f"Generated large coverage file with {len(df)} entries")
    print(f"Cells: {df['cell'].nunique()} (cell1 to cell1000)")
    print(f"Positions: {df['pos'].min()} to {df['pos'].max()}")
    print(f"Coverage range: {df['coverage'].min()} to {df['coverage'].max()}")
    print(f"Sample data:")
    print(df.head(15))

if __name__ == "__main__":
    generate_large_coverage()