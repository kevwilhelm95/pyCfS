# pyCfS
The aggregation of Lichtarge Lab genotype-phenotype validation experiments

#### Installation
pip install git+https://github.com/kevwilhelm95/pyCfS.git

#### Examples
See "example.ipynb" for help

#### Parallelization notes
Parallelized functions require the user to run function under blocking guard (i.e. if __name__ == "__main__":)

# Modules

## pyCFS.Combine

### `consensus()`
Combines multiple lists of genes, counts occurrences of each gene, and tracks the lists they came from.
#### Parameters:
- `genes_1` (list): List of genes.
- `genes_2` (list): List of genes.
- **Optional**:
  - `genes_3` (list): Additional list of genes.
  - `genes_4` (list): Additional list of genes.
  - `genes_5` (list): Additional list of genes.
  - `genes_6` (list): Additional list of genes.
  - `list_names` (list): Names of your lists (Default set to "set_x").
  - `plot_fontface` (str): Font for upset plot (Default = Avenir).
  - `plot_fontsize` (int): Font size for upset plot (Default = 14).
  - `savepath` (str): Path to save files. If no path provided, files are not saved.
#### Returns:
- `pd.DataFrame`: A dataframe with columns 'gene', 'occurrences', and 'lists', detailing each unique gene, the number of its occurrences, and the lists it appeared in.
- `Image`: Upset plot showing overlap between input genelists.
#### Parallelized:
No

### `functional_clustering()`
Clusters genes from multiple sources in STRING network 
#### Parameters:
- `genes_1` (list): list of genes
- `genes_2` (list): list of genes
- **Optional**:
    - `genes_3` (list): Additional list of genes.
    - `genes_4` (list): Additional list of genes.
    - `genes_5` (list): Additional list of genes.
    - `source_names` (list): Gene list names.  
    - `evidences` (list): Evidences to compute edge weight. Options include ['neighborhood', 'fusion', 'coocurence', 'coexpression', 'experimental', 'database', 'textmining'] (Default = ['all']).
    - `edge_confidence` (str): Minimum edge weight for network. Options include 'all' (weight > 0), 'low' (weight > 0.2), 'medium' (weight > 0.4), 'high' (weight > 0.7), 'highest' (weight > 0.9). (Default = 'highest').
    - `random_iter` (int): # of random iterations to perform (Default = 100).
    - `inflation` (float): Set inflation parameter. If not set, algorithm determines optimal inflation from 1.5-3.0.
    - `pathways_min_group_size` (int): Minimum group size for pathway enrichment (Default = 5).
    - `pathways_max_group_size` (int): Maximum group size for pathway enrichment (Default = 100).
    - `cores` (int): # of cores for parallelization (Default = 1).
    - `savepath` (str): File path. If not provided, no files are saved
#### Returns:
- `pd.DataFrame` : Pairwide edges from true connection network. Formatted as STRING network.
- `pd.DataFrame` : Table of query genes, their true clusters, and gene sources.
- `dict` : Dictionary of dataframes by cluster contained pathway enrichment results.
#### Parallelized:
Yes

### `statistical_combination()`
Statistical p-value combination methods
#### Parameters
- `df_1` (pd.DataFrame): Two-column df with genes (col1) and p-value (col2). No specific header format.
- `df_2` (pd.DataFrame): Two-column df with genes (col1) and p-value (col2).
- **Optional**:
    - `df_3` (pd.DataFrame): Additional two-column df.
    - `df_4` (pd.DataFrame): Additional two-column df.
    - `df_5` (pd.DataFrame): Additional two-column df.
    - `df_6` (pd.DataFrame): Additional two-column df.
    - `savepath` (str): File path. No files saved if not provided.
#### Returns:
- `pd.DataFrame` : Dataframe containing genes, their original p-values, and p-value combinations for Cauchy, MinP, CMC, MCM, and multiplied p-values.
#### Parallelized:
No


## pyCFS.GoldStandards
### `goldstandard_overlap()`
#### Parameters:
- `query` (list): List of query genes
- `goldstandard` (list): List of gold standard genes
- **Optional**:
    - `plot_query_color` (str): Color of query venn diagram (Default = red).
    - `plot_goldstandard_color` (str): Color of goldstandard venn diagram (Default = gray)
    - `plot_fontsize` (int): Fontsize for venn diagram (Default = 14).
    - `plot_fontface` (str): Fontface for venn diagram (Default = Avenir).
    - `savepath` (str): File path. If not provided, no files saved.
#### Returns:
- `list` : List of overlapping genes
- `float` : P-value of hypergeometric overlap.
- `Image` : Venn diagram of overlap
#### Parallelized:
No

### `ndiffusion()`

### `interconnectivity()`
#### Parameters:
- `set_1` (list): List of genes.
- `set_2` (list): List of genes. 
- **Optional**:
    - `set_3` (list): List of genes.
    - `set_4` (list): List of genes.
    - `set_5` (list): List of genes.
    - `evidences` (list): Evidences to compute edge weight. Options include ['neighborhood', 'fusion', 'coocurence', 'coexpression', 'experimental', 'database', 'textmining'] (Default = ['all']).
    - `edge_confidence` (str): Minimum edge weight for network. Options include 'all' (weight > 0), 'low' (weight > 0.2), 'medium' (weight > 0.4), 'high' (weight > 0.7), 'highest' (weight > 0.9). (Default = 'highest').
    - `num_iterations` (int): # of iterations for background connections (Default = 250). 
    - `cores` (int): For parallelization (Default = 1).
    - `plot_fontface` (str): (Default = Avenir).
    - `plot_fontsize` (int): (Default = 14).
    - `plot_query_color` (str): Line color for enrichment plot (Default = red).
    - `plot_background_color` (str): Distribution color for enrichment plot (Default = gray).
    - `savepath` (str): File path.
#### Returns:
- `Image` : Venn diagram of list overlap.
- `Image` : Random connection histogram with true query connections.
- `list` : List of random connections.
- `pd.DataFrame` : Unique gene network.
- `dict` : Gene sources
#### Parallelized:
Yes

### `gwas_catalog_colocalization()`
#### Parameters:
- `query` (list): List of genes:
- **Optional**:
    - `mondo_id` (str): Disease ID from GWAS Catalog. If used, calls API for disease.
    - `gwas_summary_path` (str): Path to GWAS Catalog downloaded summary statistics. If used, API is not called.
    - `gwas_p_thresh` (float): Filter for genome-wide significance to compare to (Default = 5e-8).
    - `distance_mbp` (float): Distance threshold for colocalization in Mbp (Default = 0.5 Mbp).
    - `cores` (int): # of cores for parallelization (Default = 1).
    - `savepath` (str): Path to save. If not used, no files are saved.
    - `save_summary_statistics` (bool): True to save downloaded summary stats if savepath is also defined.
#### Returns:
- `pd.DataFrame` : Two-column table of SNPs that colocalize with query gene.
- `float` : Fisher's exact test p-value
#### Parallelized:
Yes

### `pubmed_comentions()`
#### Parameters:
- `query` (list): List of genes.
- `keyword` (str): Keyword to search co-mentions for.
- **Optional**:
    - `email` (str): Email of a PubMed account. (Default = my email).
    - `api_key` (str): API key of PubMed account. (Default = my key).
    - `enrichment_trials` (int): # of randomization trials (Default = 100).
    - `run_enrichment` (bool): Toggle use of randomization. If False, no enrichment performed (Default = True).
    - `workers` (int): # of workers for threaded API calls (Default = 15).
    - `enrichment_cutoffs` (list): Breaks for enrichment of co-mentions (>1st number, <=2nd number; Default = [[-1,0], [0,5], [5,15], [15,50], [50,10000]])
    - `plot_background_color` (str): Color for background distribution (Default = gray).
    - `plot_query_color` (str): Color for query line (Default = red).
    - `plot_fontface` (str): Default = Avenir.
    - `plot_fontsize` (int): Default = 14.
    - `savepath` (str): File path.
#### Returns:
- `pd.DataFrame` : Dataframe of genes, co-mention counts, and PMIDs.
- `dict` : Keys = enrichment_cutoffs values; Values = (# of query genes, z score).
- `dict` : Keys = enrichment_cutoffs values; Values = Image of enrichment distribution.
#### Parallelization:
Yes



## pyCFS.Clinical
### `mouse_phenotype_enrichment()`
#### Parameters:
- `query` (list): List of genes
- **Optional**:
    - `background` (str): Background gene set. Options include 'ensembl', 'Reactomes' (Default = ensembl).
    - `random_iter` (int): Iterations for background run (Default = 5000).
    - `plot_sig_color` (str): Color for sig. phenotypes in strip plot (Default = red).
    - `plot_q_threshold` (float): Significance threshold for strip plot (Default = 0.05).
    - `plot_show_labels` (bool): If true, plot labels provided in `plot_labels_to_show` (Default = False).
    - `plot_labels_to_show` (list): Phenotype labels to plot. Use labels in "modelPhenotypeLabel" of output dataframe.
    - `cores` (int): Default = 1.
    - `savepath` (str): File path.
#### Returns:
- `pd.DataFrame` : Enrichment summary.
- `Image` : Strip plot of enrichment.
- `Image` : Histogram of FDR values.
#### Parallelized:
Yes

### `protein_family_enrichment()`
### `tissue_expression_enrichment()`
### `depmap_enrichment()`
### `drug_targets()`
