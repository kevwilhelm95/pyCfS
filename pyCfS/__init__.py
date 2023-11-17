"""
init file for pyCFS
"""
from .Combine import consensus, functional_clustering, statistical_combination
from .GoldStandards import goldstandard_overlap, interconnectivity, gwas_catalog_colocalization, pubmed_comentions
from .Clinical import mouse_phenotype_enrichment

__version__ = '0.0.2'
__author__ = 'Kevin Wilhelm, Jenn Asmussen'
