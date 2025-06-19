from setuptools import setup, find_packages

setup(
    name='snapatac_tools',
    version='0.1.0',
    description='Single-cell ATAC-seq variant calling and integration with Scanpy/AnnData',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'anndata',
        'scanpy',
        'scipy'
    ],
    python_requires='>=3.7',
)
