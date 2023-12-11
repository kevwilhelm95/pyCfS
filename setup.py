"""
Setup script for pyCfS, a package for gene list validation experiments
"""
import setuptools

VERSION = '0.0.4'

setuptools.setup(
    name = 'pyCfS',
    version = VERSION,
    url = "",
    author = "Kevin Wilhelm, Jenn Asmussen, Andrew Bigler",
    author_email = "kevin.wilhelm@bcm.edu, jennifer.asmussen@bcm.edu, andrew.bigler@bcm.edu",
    description = "Gene list validation experiments",
    long_description = open('DESCRIPTION.rst').read(),
    packages = setuptools.find_packages(),
    install_requires = ['requests', 'pandas', 'numpy', 'matplotlib', 'matplotlib_venn', 'Pillow',
         'venn', 'scipy', 'networkx', 'biopython', 'upsetplot', 'markov_clustering', 'statsmodels', 'pyarrow',
         'adjustText', 'seaborn'
    ],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ],
    include_package_data = True,
    package_data = {'':['data/*.feather', 'data/*.txt', 'data/*.gmt', 'data/*.csv', 'data/mousePhenotypes/*.parquet', 'data/targets/*.parquet']}
)
