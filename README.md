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
|consensus()    |1. genes_1 (list): list of genes to be combined and analyzed<br>2. genes_2 (list): list of genes to be combined and analyzed | 1. genes_3 (list)<br>2. genes_4 (list)<br>3. genes_5 (list)<br>4. genes_6 (list)<br>5. list_names (list): names of your lists (Default set to "set_x")<br>6. plot_fontface (str): Font for upset plot (Default = Avenir)<br>7. plot_fontsize (int): Font size for upset plot (Default = 14)<br>8. savepath (str): Path to save files. If no path provided, files are not saved.| 1. pd.DataFrame: A dataframe with columns 'gene', 'occurrences', and 'lists' detailing each unique gene, the number of its occurrences, and the lists it appeared in.<br>2. Image: Upset plot showing overlap between input genelists.| No|

### pyCFS.GoldStandards


### pyCFS.Clinical