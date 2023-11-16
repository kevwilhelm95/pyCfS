import setuptools

VERSION = '0.0.1'

setuptools.setup(
    name = 'pyCFS',
    version = VERSION,
    url = "",
    author = "Kevin Wilhelm, Jenn Asmussen, Andrew Bigler",
    author_email = "kevin.wilhelm@bcm.edu, jennifer.asmussen@bcm.edu, andrew.bigler@bcm.edu",
    description = "Gene list validation experiments",
    long_description = open('DESCRIPTION.rst').read(),
    packages = setuptools.find_packages(),
    install_requires = ['pkg_resources', 'io', 'requests', 'time', 'multiprocessing', 'typing',
        'collections', 'urllib', 'http', 'pandas', 'numpy', 'matplotlib', 'matplotlib_venn', 'PIL',
         'venn', 'scipy', 'networkx', 'Bio', 'concurrent', 'itertools', 'upsetplot', 'ast',
         'markov_clustering', 'statsmodels'
    ],
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8'
    ],
    include_package_data = True,
    package_data = {'':['data/*.feather', 'data/*.txt']}
)