# pyCfS
The aggregation of Lichtarge Lab genotype-phenotype validation experiments

#### Installation
pip install git+https://github.com/kevwilhelm95/pyCfS.git

#### Parallelization notes
Parallelized functions require the user to run function under blocking guard (i.e. if __name__ == "__main__":)

## Modules
### pyCFS.Combine
|   Function    |   Required arguments |   Optional parameters |   Returns  |   Parallelized?   |
|---------------|----------------------|-----------------------|------------|-------------------|
|consensus()    |1. genes_1 (list): list of genes<br>2. genes_2 (list): list of genes| 1. genes_3 (list)<br>2. genes_4 (list)<br>3. genes_5 (list)<br>4. genes_6 (list)<br>5. list_names (list): names of your lists (Default set to "set_x")<br>6. plot_fontface (str): Font for upset plot (Default = Avenir)<br>7. plot_fontsize (int): Font size for upset plot (Default = 14)<br>8. savepath (str): Path to save files. If no path provided, files are not saved.| 1. pd.DataFrame: A dataframe with columns 'gene', 'occurrences', and 'lists' detailing each unique gene, the number of its occurrences, and the lists it appeared in.<br>2. Image: Upset plot showing overlap between input genelists.| No|
|functional_clustering | 1. genes_1 (list): list of genes<br>2. genes_2 (list): list of genes | 1. genes_3 (list)<br>2. genes_4 (list)<br>3. genes_5 (list)<br>4. source_names (list): names of the gene lists<br>5. evidences (list): Evidences for computing edge weight (Default = ['all']. Components = [neighborhood, fusion, cooccurence, coexpression, experimental, database, textmining]).<br>6. edge_confidence (str): Minimum edge weight for network (Default = 'highest'. Options = all (>0), low (>0.2), medium (>0.4), high (>0.7), highest (>0.9)).<br>7. random_iter (int): Number of random iterations to perform for background (Default = 100)<br>8. inflation (float): Set inflation parameter. If not set, algorithm will determine optimal inflation from 1.5-3.0.<br>9.pathways_min_group_size (int): Minimum group size for functional enrichment (Default = 5).<br>10. pathways_max_group_size (int): Max group size for functional enrichment (Default = 100).<br>11. cores (int): Number of cores for parallelization (Default = 1).<br>11. savepath (str): Path to save. If not provided, no files will be saved.| 1. pd.DataFrame: Pairwise edges for true connection network.<br>2. pd.DataFrame: Table of genes, clusters, and their source.<br>3. dict: Diction of dataframes by cluster containing functional enrichment. | Yes|

### pyCFS.GoldStandards


### pyCFS.Clinical