# pyCfS
The aggregation of Lichtarge Lab genotype-phenotype validation experiments

#### Installation
pip install git+https://github.com/kevwilhelm95/pyCfS.git

#### Parallelization notes
Parallelized functions require the user to run function under blocking guard (i.e. if __name__ == "__main__":)

## Modules
### pyCFS.Combine
|   Function    |   Required arguments |   Optional parameters |   Returns  |   Parallelized?   |
|---------------|----------------------|-----------------------|-------------------|
|consensus()    |genes_1 (list): list of genes to be combined and analyzed\n genes_2 (list): list of genes to be combined and analyzed | genes_3 (list)\ngenes_4 (list)\ngenes_5 (list)\ngenes_6 (list)\nlist_names (list): names of your lists (Default set to "set_x")\nplot_fontface (str): Font for upset plot (Default = Avenir)\nplot_fontsize (int): Font size for upset plot (Default = 14)\n savepath (str): Path to save files. If no path provided, files are not saved.| 1. pd.DataFrame: A dataframe with columns 'gene', 'occurrences', and 'lists' detailing each unique gene, the number of its occurrences, and the lists it appeared in.\n2. Image: Upset plot showing overlap between input genelists.| No|

### pyCFS.GoldStandards


### pyCFS.Clinical