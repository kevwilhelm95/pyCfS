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
|   Function    |   Required arguments |   Optional parameters |   Returns  |   Parallelized?   |
|---------------|----------------------|-----------------------|------------|-------------------|
| goldstandard_overlap| 1. query(list): list of query genes<br>2. goldstandard(list): list of goldstandard genes | 1. plot_query_color(str): Color of venn diagram (Default = red)<br>2. plot_goldstandard_color (str): Color of venn diagram (Default = gray)<br>3. plot_fontsize (int): Fontsize for venn diagram (Default = 14)<br>4. plot_fontface (str): Fontface for Venn Diagram (Default = Avenir)<br>5. savepath (str): Path for saving. If not provided, no files are saved.|1. list: list of overlapping genes.<br>2. float: hypergeometric pvalue of overlap<br>3. image: Venn Diagram of overlap.| No|
|ndiffusion | Coming | Coming | Coming | Coming|
|interconnectivity | 1. set_1 (list): list of genes.<br>2. set_2 (list): list of genes.| 1. set_3 (list)<br>2. set_4 (list)<br>3. set_5 (list)<br>3. evidences (list): Evidences for computing edge weight (Default = ['all']. Components = [neighborhood, fusion, cooccurence, coexpression, experimental, database, textmining]).<br>4. edge_confidence (str): Minimum edge weight for network (Default = 'highest'. Options = all (>0), low (>0.2), medium (>0.4), high (>0.7), highest (>0.9)).<br>5. num_iterations (int): # of iterations for background connections (Default = 250).<br>6. cores (int): cores for parallelization (Default = 1).<br>7. plot_fontface (str): Default = Avenir.<br>8. plot_fontsize (int): Default = 14.<br>9. plot_background_color (str): Histogram color for enrichment (Default = gray).<br>10. plot_query_color (str): Line color for enrichment (Default = red).<br>11. savepath (str).| 1. Image: Venn diagram of overlap.<br>2. Image: Hisogram of randomized connections v. query connections.<br>3. list: list of random connections.<br>4. pd.DataFrame: Unique gene network.<br>5. dict: gene_sources.|Yes|
|gwas_catalog_colocalization | 1. query (list): list of genes. | 1. mondo_id (str): ID from GWAS Catalog of disease. If used, calls API.<br>2. gwas_summary_path (str): Path to GWAS Catalog downloaded summary statistics. If used, API is not called.<br>3. gwas_p_thresh (float): Filter for GWAS Catalog loci to compare to. (Default = 5e-8).<br>4. distance_mbp (float): Distance threshold for colocalization in Mbp (Default = 0.5 Mbp).<br>5. cores (int): # of cores for parallelized background (Default = 1).<br>6. savepath (str): Path to save.<br>7. save_summary_statistics (bool): Save downloaded summary stats if savepath is also defined.| 1. pd.DataFrame: ['Gene', 'SNP'] detailing snps that colocalize with a gene.<br>2. float: Fisher's exact test of colocalization | Yes |
|pubmed_comentions | 1. query (list): List of genes.<br>2. keyword (str): Keyword to search co-mentions for. | 1. email (str): Email of PubMed account (Default is my extended account)<br>2. api_key (str): API key of PubMed account.<br>3. enrichment_trials (int): # of randomiztion trials (Default = 100).<br>4. workers (int): # of workers for threaded API calls (Default = 15).<br>5. run_enrichment (bool): Toggle running background enrichment (Default  True).<br>6. enrichment_cutoffs (list): Breaks for determining enrichment (>1st number, <=2nd number; Default = [[-1,0], [0,5], [5,15], [15,50], [50,10000]]).<br>7. plot_background_color (str): Color for background distribution (Default = gray).<br>8. plot_query_color (str): Color for true observation (Default = red).<br>8. plot_fontface (str): Default = Avenir.<br>9. plot_fontsize (int): Default = 14.<br>10. savepath (str): Path to save | 1. pd.DataFrame: Dataframe of genes, co-mention counts, and PMIDs.<br>2. dict: Keys = enrichment_cutoff values; Values = (# of query genes, z_score).<br>3. dict: Keys = Enrichment_cutoff values; Values = Image of enrichment distribution | Yes |

### pyCFS.Clinical
|   Function    |   Required arguments |   Optional parameters |   Returns  |   Parallelized?   |
|---------------|----------------------|-----------------------|------------|-------------------|
| mouse_phenotype_enrichment |1. query (list): List of genes | 1. background (str): Background gene set (Default = ensembl. Options = ensembl, Reactomes)<br>2. random_iter (int): Iterations for background runs (Default = 5000).<br>3. plot_sig_color (str): Color for strip_plot significant genes (Default = red).<br>4. plot_q_threshold (float): Significance threshold for strip_plot (Default = 0.05).<br>5. plot_show_labels (bool): Choose to plot labels. If true, must also provide labels in plot_labels_to_show (Default = False).<br>6. plot_labels_to_show (list): Phenotype labels to plot. Use "modelPhenotypeLabel" in output dataframe.<br>7. cores (int): Default = 1.<br>7. savepath (str): Path to save. No files saved if empty. | 1. pd.DataFrame: Summary of enrichments.<br>2. Image: Strip plot of enrichment.<br>3. Image: Histogram of FDR values. | Yes|
| protein_family_enrichment | Coming | Coming | Coming | Coming |
| tissue_expression_enrichment | Coming | Coming | Coming | Coming|
| depmap_enrichment | Coming | Coming | Coming | Coming|
| drug_targets | Coming | Coming | Coming | Coming |
