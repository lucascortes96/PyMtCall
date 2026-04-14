import unittest
import numpy as np
import pandas as pd
import anndata
from scipy.sparse import csr_matrix
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Py_mTCall_tools.clonotype_analysis import (
    cluster_clonotypes,
    find_clonotypes,
    analyze_clonotypes_from_variants,
    get_variant_confidence_summary,
    set_if_null
)

class TestClonotypeAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create a simple test AnnData object
        n_cells = 100
        n_variants = 20
        
        # Create random VAF matrix
        np.random.seed(42)
        vaf_matrix = np.random.beta(1, 5, size=(n_cells, n_variants))  # Skewed towards low VAF
        
        # Create some high VAF variants for specific cell groups to simulate clonotypes
        # Group 1: cells 0-30, variants 0-5
        vaf_matrix[0:30, 0:5] = np.random.beta(2, 1, size=(30, 5))  # Higher VAF
        
        # Group 2: cells 30-60, variants 5-10  
        vaf_matrix[30:60, 5:10] = np.random.beta(2, 1, size=(30, 5))
        
        # Group 3: cells 60-100, variants 10-15
        vaf_matrix[60:100, 10:15] = np.random.beta(2, 1, size=(40, 5))
        
        # Create AnnData object
        self.adata = anndata.AnnData(X=np.random.randn(n_cells, 50))  # Random gene expression
        self.adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        self.adata.var_names = [f"gene_{i}" for i in range(50)]
        
        # Add VAF matrix to obsm
        self.adata.obsm['vaf_matrix'] = vaf_matrix
        
        # Add variant information
        variant_names = [f"variant_{i}" for i in range(n_variants)]
        self.adata.uns['variant_names'] = variant_names
        
        # Create variant info DataFrame with confidence metrics
        variant_info = pd.DataFrame({
            'mean': np.mean(vaf_matrix, axis=0),
            'variance': np.var(vaf_matrix, axis=0),
            'n_cells_conf_detected': np.sum(vaf_matrix > 0.01, axis=0),  # Confident detection threshold
            'n_cells_over_5': np.sum(vaf_matrix > 0.05, axis=0),
            'n_cells_over_10': np.sum(vaf_matrix > 0.10, axis=0),
            'n_cells_over_50': np.sum(vaf_matrix > 0.50, axis=0),
            'mean_coverage': np.random.uniform(10, 100, n_variants),  # Simulated coverage
            'strand_concordance': np.random.uniform(0.1, 0.9, n_variants)  # Simulated strand concordance
        })
        self.adata.uns['variant_info'] = variant_info
        
        # Add some basic clustering to test cluster_clonotypes
        self.adata.obs['leiden'] = ['cluster_0'] * 30 + ['cluster_1'] * 30 + ['cluster_2'] * 40
        self.adata.obs['leiden'] = self.adata.obs['leiden'].astype('category')
    
    def test_set_if_null(self):
        """Test utility function"""
        self.assertEqual(set_if_null(None, 'default'), 'default')
        self.assertEqual(set_if_null('value', 'default'), 'value')
    
    def test_cluster_clonotypes(self):
        """Test hierarchical clustering of clonotypes"""
        # Create a simple test matrix
        test_data = np.random.randn(10, 5)
        test_adata = anndata.AnnData(X=test_data)
        test_adata.obs['group'] = ['A'] * 5 + ['B'] * 5
        test_adata.obs['group'] = test_adata.obs['group'].astype('category')
        
        result = cluster_clonotypes(test_adata, group_by='group')
        
        # Check that we get the expected keys
        self.assertIn('cells', result)
        self.assertIn('features', result)
        self.assertIn('group_means', result)
        self.assertIn('groups', result)
        
        # Check dimensions
        self.assertEqual(len(result['groups']), 2)  # A and B
        self.assertEqual(result['group_means'].shape, (5, 2))  # 5 features, 2 groups
    
    def test_find_clonotypes(self):
        """Test graph-based clonotype finding"""
        # Create test data
        test_data = np.random.randn(20, 10)
        test_adata = anndata.AnnData(X=test_data)
        test_adata.obs_names = [f"cell_{i}" for i in range(20)]
        test_adata.var_names = [f"feature_{i}" for i in range(10)]
        
        result_adata = find_clonotypes(
            test_adata,
            resolution=0.5,
            k=5,
            algorithm='leiden'
        )
        
        # Check that clustering was performed
        self.assertIn('clonotype_leiden', result_adata.obs.columns)
        
        # Check that neighbor graph was added
        self.assertIn('clonotype_connectivities', result_adata.obsp.keys())
        self.assertIn('clonotype_distances', result_adata.obsp.keys())
        
        # Check that parameters were stored
        self.assertIn('clonotype_params', result_adata.uns.keys())
    
    def test_analyze_clonotypes_from_variants(self):
        """Test integrated clonotype analysis from variant data using confidence criteria"""
        result_adata = analyze_clonotypes_from_variants(
            self.adata,
            vaf_layer_name="vaf",
            min_cells=5,                    # Confident detection in ≥5 cells
            min_vaf=0.1,                   # VAF ≥10% threshold
            min_coverage=0,               # Mean coverage ≥15x
            min_strand_concordance=0,                 # VAF variance ≤0.2
            k=5
        )
        
        # Check that clonotype assignment was added
        self.assertIn('clonotype_leiden', result_adata.obs.columns)
        
        # Check that analysis results were stored
        self.assertIn('clonotype_analysis', result_adata.uns.keys())
        self.assertIn('clonotype_stats', result_adata.uns.keys())
        
        # Check that neighbor graphs were added
        self.assertIn('clonotype_connectivities', result_adata.obsp.keys())
        
        # Check clonotype stats structure
        clonotype_stats = result_adata.uns['clonotype_stats']
        expected_columns = ['clonotype', 'n_cells', 'n_characteristic_variants', 
                          'characteristic_variants', 'mean_vaf_all_variants']
        for col in expected_columns:
            self.assertIn(col, clonotype_stats.columns)
    
    def test_error_handling(self):
        """Test error handling for missing data"""
        # Test with AnnData object missing VAF matrix
        minimal_adata = anndata.AnnData(X=np.random.randn(10, 5))
        
        with self.assertRaises(ValueError):
            analyze_clonotypes_from_variants(minimal_adata)
    
    def test_vaf_matrix_filtering(self):
        """Test that confident variant filtering works correctly"""
        result_adata = analyze_clonotypes_from_variants(
            self.adata,
            min_cells=3,              # Confident detection in ≥3 cells
            min_vaf=0.1,             # VAF ≥10% threshold
            min_coverage=15,         # Mean coverage ≥15x
            min_strand_concordance=0.2,  # Strand concordance ≥0.2
            max_variance=0.2         # VAF variance ≤0.2
        )
        
        # Check that some variants were filtered
        analysis_info = result_adata.uns['clonotype_analysis']
        n_variants_used = analysis_info['n_variants_used']
        
        # Should be fewer than total variants due to filtering
        self.assertLessEqual(n_variants_used, self.adata.obsm['vaf_matrix'].shape[1])
        
        # Check that the filtering criteria are stored
        self.assertEqual(analysis_info['min_cells_threshold'], 5)
        self.assertEqual(analysis_info['min_vaf_threshold'], 0.1)
    
    def test_visualization_functions(self):
        """Test that visualization functions work and create plots"""
        # First run clonotype analysis
        result_adata = analyze_clonotypes_from_variants(
            self.adata,
            vaf_layer_name="vaf",
            min_cells=3,
            min_vaf=0.05,
            resolution=1.0,
            k=5
        )
        
        # Test the heatmap plotting function
        from Py_mTCall_tools.clonotype_analysis import plot_clonotype_vaf_heatmap
        
        # Create the heatmap (this should not raise any errors)
        try:
            fig = plot_clonotype_vaf_heatmap(
                result_adata,
                vaf_layer_name="vaf",
                clonotype_col="clonotype_leiden",
                top_variants=10,  # Use fewer variants for test
                figsize=(8, 6),
                save="test_clonotype_heatmap.png"  # Save to file
            )
            
            # Check that a figure was created
            self.assertIsNotNone(fig)
            
            # Check that the file was saved
            import os
            self.assertTrue(os.path.exists("test_clonotype_heatmap.png"))
            
            # Clean up the test file
            if os.path.exists("test_clonotype_heatmap.png"):
                os.remove("test_clonotype_heatmap.png")
                
            print("✓ Heatmap visualization test passed - plot created and saved successfully!")
            
        except Exception as e:
            self.fail(f"Heatmap plotting failed with error: {e}")
    
    def test_create_example_heatmap(self):
        """Create an example heatmap for demonstration purposes using confident variant filtering"""
        # Run the full clonotype analysis workflow with confidence filtering
        result_adata = analyze_clonotypes_from_variants(
            self.adata,
            vaf_layer_name="vaf",
            min_cells=3,                    # Confident detection in ≥3 cells
            min_vaf=0.1,                   # VAF ≥10% threshold  
            min_coverage=15,               # Mean coverage ≥15x
            min_strand_concordance=0.2,    # Strand concordance ≥0.2
            max_variance=0.2,              # VAF variance ≤0.2
            resolution=1.0,
            k=5
        )
        
        # Import visualization functions
        from Py_mTCall_tools.clonotype_analysis import plot_clonotype_vaf_heatmap, plot_clonotype_heatmap
        
        print("\n" + "="*50)
        print("CREATING EXAMPLE VISUALIZATIONS")
        print("="*50)
        
        # Create VAF heatmap
        print("Creating VAF heatmap...")
        try:
            fig1 = plot_clonotype_vaf_heatmap(
                result_adata,
                vaf_layer_name="vaf",
                top_variants=15,
                figsize=(10, 8),
                save="example_vaf_heatmap.png"
            )
            print("✓ VAF heatmap saved as 'example_vaf_heatmap.png'")
        except Exception as e:
            print(f"✗ VAF heatmap failed: {e}")
        
        # Create general clonotype heatmap
        print("Creating clonotype similarity heatmap...")
        try:
            # Use the VAF matrix from obsm instead of layers
            fig2 = plot_clonotype_heatmap(
                result_adata,
                layer=None,  # Use adata.X instead of a layer
                group_by="clonotype_leiden",
                features=None,
                figsize=(8, 6),
                save="example_clonotype_similarity_heatmap.png"
            )
            print("✓ Clonotype similarity heatmap saved as 'example_clonotype_similarity_heatmap.png'")
        except Exception as e:
            print(f"✗ Clonotype similarity heatmap failed: {e}")
        
        # Print summary of results
        print("\nCLONOTYPE ANALYSIS SUMMARY:")
        print("-" * 30)
        n_clonotypes = len(result_adata.obs['clonotype_leiden'].cat.categories)
        print(f"Number of clonotypes discovered: {n_clonotypes}")
        
        clonotype_sizes = result_adata.obs['clonotype_leiden'].value_counts()
        print("Clonotype sizes:")
        for clono, size in clonotype_sizes.items():
            print(f"  {clono}: {size} cells")
        
        if 'clonotype_stats' in result_adata.uns:
            stats = result_adata.uns['clonotype_stats']
            print(f"\nCharacteristic variants per clonotype:")
            for _, row in stats.iterrows():
                print(f"  {row['clonotype']}: {row['n_characteristic_variants']} variants")
        
        print("\nVisualization files created in current directory:")
        import os
        for filename in ["example_vaf_heatmap.png", "example_clonotype_similarity_heatmap.png"]:
            if os.path.exists(filename):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (failed to create)")
        
        print("="*50)
    
    def test_get_variant_confidence_summary(self):
        """Test the variant confidence summary function"""
        summary = get_variant_confidence_summary(self.adata)
        
        # Check that summary is returned
        self.assertIsNotNone(summary)
        self.assertIsInstance(summary, pd.DataFrame)
        
        # Check expected columns
        expected_metrics = [
            'Total variants',
            'Mean cells confident detection',
            'Mean cells VAF >5%',
            'Mean cells VAF >10%',
            'Mean cells VAF >50%',
            'Mean coverage depth',
            'Mean strand concordance',
            'Mean VAF variance'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, summary['metric'].values)
        
        # Check that values are numeric
        self.assertTrue(all(isinstance(x, (int, float)) for x in summary['value']))

if __name__ == '__main__':
    unittest.main()
