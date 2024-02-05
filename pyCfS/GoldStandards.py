"""Collection of functions looking at previous genetic overlap recovery

Functions:

"""

import pkg_resources
import io
import os
import requests
import time
from multiprocessing import Pool
from typing import Any
from urllib.error import HTTPError
from http.client import IncompleteRead
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from PIL import Image
from venn import venn
from scipy.stats import norm, fisher_exact
import networkx as nx
from Bio import Entrez
import concurrent.futures
from itertools import repeat
from .utils import _hypergeo_overlap, _format_scientific, _fix_savepath, _clean_genelists, _load_grch38_background, _load_string, _get_evidence_types, _get_combined_score, _get_edge_weight, _select_evidences

#region Gold Standard Overlap
def _get_overlap(list1:list, list2:list) -> (list, float):
    """
    Calculates the overlap between two lists and computes the p-value for the overlap.

    This function identifies the common elements between two lists and calculates the
    statistical significance (p-value) of this overlap using a hypergeometric test.
    The background gene set for the hypergeometric test is loaded from a predefined
    function `_load_grch38_background`.

    Args:
        list1 (list): The first list, typically representing a set of query genes.
        list2 (list): The second list, usually representing a gene set (e.g., from a gene set database).

    Returns:
        tuple:
            - list: A list containing the overlapping elements between `list1` and `list2`.
            - float: The p-value indicating the statistical significance of the overlap,
                     calculated using the hypergeometric test.
    """
    # Get overlap and calculate p-val
    overlap = [x for x in list1 if x in list2]
    pval = _hypergeo_overlap(background_size = len(_load_grch38_background()), query_genes = len(list1), gs_genes = len(list2), overlap = len(overlap))

    return overlap, pval

def _plot_overlap_venn(query_len:int, goldstandard_len:int, overlap:list, pval:float, show_genes_pval: bool, query_color:str, goldstandard_color:str, fontsize:int, fontface:str) -> Image:
    """
    Plots a Venn diagram representing the overlap between two sets and returns the plot as an image.

    This function creates a Venn diagram to visualize the overlap between a query set and a gold standard set.
    It displays the overlap size, the p-value of the overlap, and the names of the overlapping items. If there
    are no overlapping genes or if the query set is empty, the function will print a relevant message and return False.

    Args:
        query_len (int): The number of elements in the query set.
        goldstandard_len (int): The number of elements in the gold standard set.
        overlap (list): A list of overlapping elements between the query and gold standard sets.
        pval (float): The p-value representing the statistical significance of the overlap.
        query_color (str): The color to be used for the query set in the Venn diagram.
        goldstandard_color (str): The color to be used for the gold standard set in the Venn diagram.
        fontsize (int): The font size to be used in the Venn diagram.
        fontface (str): The font face to be used in the Venn diagram.

    Returns:
        Image: An image object of the Venn diagram. If there is no overlap or the query is empty, returns False.
    """
    if len(overlap) == 0:
        print("No overlapping genes to plot")
        return False
    elif query_len == 0:
        print("No genes found in query")
        return False
    # Create Venn Diagram
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _ = plt.figure(figsize=(10, 5))
    out = venn2(subsets=((query_len - len(overlap)),
                        (goldstandard_len - len(overlap)),
                        len(overlap)),
                        set_labels=('Query', 'Goldstandard'),
                        set_colors=('white', 'white'),
                        alpha=0.7)
    overlap1 = out.get_patch_by_id("A")
    overlap1.set_edgecolor(query_color)
    overlap1.set_linewidth(3)
    overlap2 = out.get_patch_by_id("B")
    overlap2.set_edgecolor(goldstandard_color)
    overlap2.set_linewidth(3)

    for text in out.set_labels:
        text.set_fontsize(fontsize + 2)
    for text in out.subset_labels:
        if text == None:
            continue
        text.set_fontsize(fontsize)
    if show_genes_pval:
        plt.text(0, -0.7,
                str("p = " + _format_scientific(pval)),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize-2)
        plt.text(0, -0.78,
                ", ".join(overlap),
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=fontsize-2)
    plt.title("Gold Standard Overlap", fontsize=fontsize+4)
    plt.tight_layout(pad = 2.0)
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    #buffer.close()
    plt.close()

    return image

def goldstandard_overlap(query: list, goldstandard:list, plot_query_color:str = 'red', plot_goldstandard_color:str = 'gray', plot_show_gene_pval:bool = True, plot_fontsize:int = 14, plot_fontface:str = 'Avenir', savepath:Any = False) -> (list, float, Image):
    """
    Analyzes the overlap between a query gene list and a gold standard gene list, plots a Venn diagram of the overlap,
    and optionally saves the plot and summary.

    This function first cleans the input gene lists, then calculates the overlap between the query and gold standard
    gene lists, including the p-value for this overlap. It then creates a Venn diagram to visually represent the overlap.
    If a save path is provided, the function will save the Venn diagram image and a summary text file with the overlapping
    genes and the p-value.

    Args:
        query (list): A list of genes representing the query set.
        goldstandard (list): A list of genes representing the gold standard set.
        plot_query_color (str, optional): Color for the query set in the Venn diagram. Defaults to 'red'.
        plot_goldstandard_color (str, optional): Color for the gold standard set in the Venn diagram. Defaults to 'gray'.
        plot_fontsize (int, optional): Font size for the text in the Venn diagram. Defaults to 14.
        plot_fontface (str, optional): Font face for the text in the Venn diagram. Defaults to 'Avenir'.
        savepath (Any, optional): Path to save the output files. If False, files are not saved. Defaults to False.

    Returns:
        tuple:
            - list: A list of genes that overlap between the query and gold standard sets.
            - float: The p-value indicating the statistical significance of the overlap.
            - Image: An image object of the Venn diagram. None if there is no overlap or the query set is empty.

    Note:
        The function depends on the '_clean_genelists', '_get_overlap', and '_plot_overlap_venn' functions being
        available in the same code base and correctly implemented.
    """
    cleaned = _clean_genelists([query, goldstandard])
    query, goldstandard = cleaned[0], cleaned[1]
    # Get overlap
    overlapping_genes, pval = _get_overlap(query, goldstandard)
    # Plot Venn diagram
    image = _plot_overlap_venn(len(query), len(goldstandard), overlapping_genes, pval, plot_show_gene_pval, plot_query_color, plot_goldstandard_color, plot_fontsize, plot_fontface)
    # Output files
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'GoldStandard_Overlap/')
        os.makedirs(new_savepath, exist_ok=True)
        if image:
            image.save(new_savepath + "GoldStandard_Overlap.png", bbox_inches = 'tight', pad_inches = 0.5)
        with open(new_savepath + "GoldStandard_Overlap_Summary.txt", 'w') as f:
            f.write(f"Overlapping genes: {overlapping_genes}\n")
            f.write(f"P-value: {pval}")

    return overlapping_genes, pval, image
#endregion



#region nDiffusion


def nDiffusion():
    network_fl = '../data/networks/toy_network.txt'
    geneList1_fl = '../data/genes/A.tsv'
    geneList2_fl = '../data/genes/B.tsv'
    result_fl = '../results/'
    group1_name = geneList1_fl.split('/')[-1].split('.')[0]
    group2_name = geneList2_fl.split('/')[-1].split('.')[0]
    repeat = 100 #number of randomization iterations

    ### For a multimodal network, specify graph genes
    graph_gene = []
    if network_fl == '../data/networks/MeTeOR.txt':
        for line in open(network_fl).readlines():
            line = line.strip('\n').split('\t')
            for i in line[:2]:
                if ':' not in i:
                    graph_gene.append(i)

    if __name__ == "__main__":
        ### Directory of the result folder
        result_fl_figure = result_fl + 'figures/'
        result_fl_raw = result_fl + 'raw_data/'
        result_fl_ranking = result_fl + 'ranking/'
        if not os.path.exists(result_fl):
            os.makedirs(result_fl)
        if not os.path.exists(result_fl_figure):
            os.makedirs(result_fl_figure)
        if not os.path.exists(result_fl_raw):
            os.makedirs(result_fl_raw)   
        if not os.path.exists(result_fl_ranking):
            os.makedirs(result_fl_ranking)      
        print('Running ...')

        ### Getting network and diffusion parameters
        G, graph_node, adjMatrix, node_degree, G_degree = getGraph(network_fl)
        ps = getDiffusionParam(adjMatrix)
        graph_node_index = getIndexdict(graph_node)
        GP1_only_dict, GP2_only_dict, overlap_dict, other_dict = parseGeneInput(geneList1_fl, geneList2_fl, graph_node, graph_node_index, node_degree, graph_gene)
        degree_nodes = getDegreeNode(G_degree, node_degree, other_dict['node'])

        # Combine exclusive genes and overlapped genes in each group, if there is an overlap
        if overlap_dict['node'] != []:
            GP1_all_dict = combineGroup(GP1_only_dict, overlap_dict)
            GP2_all_dict = combineGroup(GP2_only_dict, overlap_dict)
            Exclusives_dict = combineGroup(GP1_only_dict, GP2_only_dict)
    
        ### Diffusion experiments
        def getResults(gp1, gp2, result_fl, gp1_name, gp2_name, show = '', exclude=[]):
            auroc, z_auc, auprc, z_prc, pval = runrun(gp1, gp2, result_fl, gp1_name, gp2_name, show, degree_nodes, other_dict['node'], graph_node_index, graph_node, ps, exclude=exclude, repeat = repeat)
            return auroc, z_auc, auprc, z_prc, pval
        
        #### auroc, z-scores for auc, auprc, z-scores for auprc, KS pvals
        #### z-scores: from_degree, to_degree, from_uniform, to_uniform

        if overlap_dict['node'] != [] and GP1_only_dict['node'] != [] and GP2_only_dict['node'] != []: 
            # From group 1 exclusive to group 2 all:
            R_gp1o_gp2 = getResults(GP1_only_dict, GP2_all_dict, result_fl, group1_name+'Excl', group2_name, show = '__SHOW_1_')
            # From group 2 exclusive to group 1 all:
            R_gp2o_gp1 = getResults(GP2_only_dict, GP1_all_dict, result_fl, group2_name+'Excl', group1_name, show = '__SHOW_2_')     
            # From group 1 exclusive to group 2 exclusive:
            R_gp1o_gp2o = getResults(GP1_only_dict, GP2_only_dict, result_fl, group1_name+'Excl', group2_name+'Excl')
            # From group 2 exclusive to group 1 exclusive:
            R_gp2o_gp1o = getResults(GP2_only_dict, GP1_only_dict, result_fl, group2_name+'Excl', group1_name+'Excl')
            # From group 1 exclusive to the overlap
            R_gp1o_overlap = getResults(GP1_only_dict, overlap_dict, result_fl, group1_name+'Excl', 'Overlap')
            # From group 2 exclusive to the overlap
            R_gp2o_overlap = getResults(GP2_only_dict, overlap_dict, result_fl, group2_name+'Excl', 'Overlap')
            # From overlap to (group 1 exclusive and group 2 exlusive)
            R_overlap_exclusives = getResults(overlap_dict, Exclusives_dict, result_fl,'Overlap', 'Exclus')
            ### Write output
            writeSumTxt (result_fl, group1_name, group2_name, GP1_only_dict, GP2_only_dict, overlap_dict, R_gp1o_gp2, R_gp2o_gp1, R_gp1o_gp2o, R_gp2o_gp1o, R_gp1o_overlap, R_gp2o_overlap, R_overlap_exclusives)
        elif overlap_dict['node'] != [] and GP2_only_dict['node'] == []: #when group 2 is entirely part of group 1
            # From group 1 exclusive to overlap/group 2
            R_gp1o_overlap = getResults(GP1_only_dict, overlap_dict, result_fl, group1_name+'Excl', 'Overlap or'+group2_name)
            # From overlap/group 2 to group 1 exclusive
            R_overlap_gp1o = getResults(overlap_dict, GP1_only_dict, result_fl,'Overlap or'+group2_name, group1_name+'Excl')
            writeSumTxt (result_fl, group1_name, group2_name, GP1_only_dict, GP2_only_dict, overlap_dict, R_gp1o_overlap=R_gp1o_overlap, R_overlap_gp1o=R_overlap_gp1o)
        elif overlap_dict['node'] != [] and GP1_only_dict['node'] == []: #when group 1 is entirely part of group 2
            # From group 2 exclusive to overlap/group 1
            R_gp2o_overlap = getResults(GP2_only_dict, overlap_dict, result_fl, group2_name+'Excl', 'Overlap or '+group1_name)
            # From overlap/group 1 to group 2 exclusive
            R_overlap_gp2o = getResults(overlap_dict, GP2_only_dict, result_fl,'Overlap or'+group1_name, group2_name+'Excl')
            writeSumTxt (result_fl, group1_name, group2_name, GP1_only_dict, GP2_only_dict, overlap_dict, R_gp2o_overlap=R_gp2o_overlap, R_overlap_gp2o=R_overlap_gp2o)
        else: #when there is no overlap between two groups
            # From group 1 to group 2:
            R_gp1o_gp2o = getResults(GP1_only_dict, GP2_only_dict, result_fl, group1_name, group2_name, show = 'SHOW1')
            # From group 2 to group 1:
            R_gp2o_gp1o = getResults(GP2_only_dict, GP1_only_dict, result_fl, group2_name, group1_name, show = 'SHOW2')
            ### Write output
            writeSumTxt (result_fl, group1_name, group2_name, GP1_only_dict, GP2_only_dict, overlap_dict, R_gp1o_gp2o=R_gp1o_gp2o, R_gp2o_gp1o=R_gp2o_gp1o)
#endregion



#region First neighbor connectivity
def _load_genelists(set_1:list, set_2:list, set_3:list, set_4:list, set_5: list) -> dict:
    """
    Loads gene lists and organizes them into a dictionary.

    Each gene list is associated with a key following the naming convention 'set1', 'set2', etc.
    Only non-None lists are included in the returned dictionary.

    Parameters
    ----------
    set_1 : list
        The first gene list.
    set_2 : list
        The second gene list.
    set_3 : list
        The third gene list.
    set_4 : list
        The fourth gene list.
    set_5 : list
        The fifth gene list.

    Returns
    -------
    dict
        A dictionary where each key corresponds to a 'set' label (e.g., 'set1') and each value
        is one of the input gene lists. Only non-None gene lists are included.

    Examples
    --------
    >>> gene_lists = _load_genelists(['gene1', 'gene2'], ['gene3', 'gene4'], None, None, None)
    >>> print(gene_lists)
    {'set1': ['gene1', 'gene2'], 'set2': ['gene3', 'gene4']}
    """
    clean_lists = _clean_genelists([set_1, set_2, set_3, set_4, set_5])
    sets = ['set'+str(i+1) for i in range(len(clean_lists))]
    gene_dict = {}
    for i in range(len(clean_lists)):
        gene_dict[sets[i]] = clean_lists[i]
    return gene_dict

def _get_gene_sources(set_dict: dict) -> dict:
    """
    Maps each gene to the sets it belongs to in a given dictionary.

    This function processes a dictionary where each key is a set (e.g., a specific condition or category) and
    the corresponding value is a list of genes (proteins) associated with that set. It inverts this relationship
    to create a new dictionary where each gene is mapped to a list of sets it belongs to.

    Args:
        set_dict (dict): A dictionary with set names as keys and lists of genes (proteins) as values.

    Returns:
        dict: A dictionary where each key is a gene (protein), and the corresponding value is a list of set names
              that the gene is associated with.

    Note:
        This function is useful in contexts where the association between genes and various sets (like experimental
        conditions, categories, or any grouping criteria) needs to be analyzed. If a gene is associated with multiple sets,
        all such associations are captured in the list corresponding to that gene.
    """
    gene_source = {}
    for k, v in set_dict.items():
        for protein in set_dict[k]:
            if protein in gene_source:
                gene_source[protein].append(k)
            else:
                gene_source[protein] = [k]
    return gene_source

def _get_unique_genes(source_dict:dict) -> dict:
    """
    Extracts unique genes from the source dictionary.

    This function iterates through the source dictionary and collects genes that
    are associated with only one source.

    Arguments:
    ---------
        source_dict (dict): A dictionary with keys as sources and values as lists of gene names.

    Returns:
    -------
        dict: A dictionary containing unique genes and their associated source.
    """
    unique_genes = {}
    for k, v in source_dict.items():
        if len(v) == 1:
            unique_genes[k] = v
    return unique_genes

def _check_connection(x:str, y:str, gene_lst:list) -> str:
    """
    Checks if both specified genes are present in a given gene list.

    This function examines whether both of the two specified genes (x and y) are present in a provided list of genes.
    It is primarily used to determine if there is a connection or association between two genes within the context of a given gene list.

    Args:
        x (str): The first gene to check for presence in the gene list.
        y (str): The second gene to check for presence in the gene list.
        gene_lst (list): A list of gene symbols (strings) to search within.

    Returns:
        str: Returns 'yes' if both genes are found in the gene list; otherwise, returns 'no'.

    Note:
        This function is useful in network analysis or genetic studies where determining the co-occurrence or connection of genes
        within a specific list or dataset is needed. It performs a simple presence check and does not infer any biological or functional relationship.
    """
    if x in gene_lst and y in gene_lst:
        out = 'yes'
    else:
        out = 'no'
    return out

def _get_node_pair(x:str, y:str) -> str:
    """
    Creates a sorted string representation of a pair of nodes (genes or proteins).

    This function takes two node identifiers (such as gene or protein symbols) and returns a string representation
    of the node pair. The two nodes are sorted alphabetically and combined into a single string, which facilitates
    consistent identification of node pairs regardless of the order in which they are provided.

    Args:
        x (str): The identifier for the first node.
        y (str): The identifier for the second node.

    Returns:
        str: A string representation of the node pair, where the node identifiers are sorted alphabetically
             and combined in a list format.

    Note:
        This function is useful in network analysis and graph operations where consistent identification of an
        edge or relationship between two nodes is required. It ensures that the pair 'x, y' is treated identically
        to the pair 'y, x', thereby avoiding duplicate representations of the same edge.
    """
    pair = [x, y]
    pair.sort()
    return str(pair)

def _get_unique_gene_network(gene_lst:list, network:pd.DataFrame) -> pd.DataFrame:
    """
    Generates a sub-network DataFrame containing only the connections between genes in a specified list.

    This function takes a list of genes and a network DataFrame, then filters and returns a new DataFrame representing
    the sub-network of connections exclusively among the genes in the list. It ensures that each connection (edge) is
    unique and appears only once in the sub-network.

    Args:
        gene_lst (list): A list of gene symbols (strings) to be included in the sub-network.
        network (pd.DataFrame): A DataFrame representing a network with columns 'node1' and 'node2',
                                indicating connections between nodes (genes or proteins).

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the connections between the genes specified in gene_lst.
                      Each row represents a unique connection, and duplicate connections are removed.

    Note:
        The function first filters the network to include connections where either 'node1' or 'node2' is in gene_lst,
        then further filters to include only connections where both nodes are in the list. It also ensures that each
        connection is represented only once, regardless of the order of nodes, by creating a sorted pair string for each
        connection and removing duplicates based on these pairs.
    """
    n_df = network.copy()
    n_df1 = n_df[n_df['node1'].isin(gene_lst)]
    n_df2 = n_df[n_df['node2'].isin(gene_lst)]
    n_df_final = pd.concat([n_df1, n_df2])
    n_df_final['bwSetConnection'] = n_df_final.apply(lambda x: _check_connection(x['node1'], x['node2'], gene_lst),
                                                     axis=1)
    n_df_final = n_df_final[n_df_final['bwSetConnection'] == 'yes']
    n_df_final.drop_duplicates(inplace=True)
    n_df_final['pair'] = n_df_final.apply(
        lambda x: _get_node_pair(x['node1'], x['node2']), axis=1)
    n_df_final.sort_values(by='node1', inplace=True)
    n_df_final.drop_duplicates(subset=['pair'], inplace=True, keep='first')
    return n_df_final

def _get_unique_gene_network_bw_method_connections(network:pd.DataFrame, unique_dict:dict) -> list:
    """
    Identifies between-method connections in a gene network.

    This function processes a network DataFrame and a dictionary of unique gene sets
    to find connections between genes that belong to different sets (methods).

    Arguments:
    ---------
        network (pd.DataFrame): A DataFrame containing the network edges with columns 'node1' and 'node2'.
        unique_dict (dict): A dictionary with keys as gene names and values as lists indicating the method/set they belong to.

    Returns:
    -------
        list: A list of unique between-method edge pairs.

    """
    net_genes = list(
        set(network['node1'].tolist() + network['node2'].tolist()))
    bw_method_edges = []

    for g in net_genes:
        df1 = network[network['node1'] == g]
        df1_genes = df1['node2'].tolist()

        for p in df1_genes:
            if unique_dict[p][0] != unique_dict[g][0]:
                bw_method_edges.append([g, p])

        df2 = network[network['node2'] == g]
        df2_genes = df2['node1'].tolist()

        for p in df2_genes:
            if unique_dict[p][0] != unique_dict[g][0]:
                bw_method_edges.append([g, p])

    bw_method_edges = [sorted(x) for x in bw_method_edges]
    bw_method_edges = [str(x) for x in bw_method_edges]
    bw_method_edges = list(set(bw_method_edges))

    return bw_method_edges

def _get_unique_gene_counts(unique_gene_dict: dict) -> dict:
    """
    Counts the occurrences of unique genes across different sets in a dictionary.

    This function processes a dictionary where each key represents a gene or a set of genes, and the value is a list
    with the first element typically being a gene name or identifier. It then creates a new dictionary where each key
    is a unique gene, and the value is a list of keys from the input dictionary where the gene is present.

    Args:
        unique_gene_dict (dict): A dictionary with keys representing gene sets or similar constructs, and values are lists,
                                 with the first element of each list being a gene name or identifier.

    Returns:
        dict: A dictionary where each key is a unique gene, and the corresponding value is a list of keys from the
              input dictionary where the gene is mentioned.

    Note:
        This function is particularly useful in situations where one needs to map each gene to the sets or categories
        it belongs to, based on a given dictionary structure. It assumes that the first element of each list in the
        values of `unique_gene_dict` is the gene identifier.
    """
    unique_genes_per_set = {}
    for k, v in unique_gene_dict.items():
        if v[0] in unique_genes_per_set:
            unique_genes_per_set[v[0]].append(k)
        else:
            unique_genes_per_set[v[0]] = [k]
    return unique_genes_per_set

def _get_node_degree_dict(unique_genes:list, degree_df:pd.DataFrame) -> pd.DataFrame:
    """
    Filters a DataFrame to include only the degrees of specified nodes (genes).

    This function takes a list of unique genes and a DataFrame containing node degrees, then filters the DataFrame
    to include only the rows corresponding to the unique genes in the list. The resulting DataFrame contains degree
    information exclusively for the specified genes.

    Args:
        unique_genes (list): A list of gene symbols (strings) whose degree information is to be extracted.
        degree_df (pd.DataFrame): A DataFrame containing degree information for nodes (genes). The DataFrame should
                                  have an index or a column that matches the gene symbols in the `unique_genes` list.

    Returns:
        pd.DataFrame: A DataFrame filtered to include only the rows corresponding to the genes in `unique_genes`.
                      The structure of the DataFrame remains the same as `degree_df`, but only with the relevant rows.

    Note:
        This function assumes that the index or a key column of `degree_df` directly corresponds to gene symbols.
        It is essential that `unique_genes` contains valid entries that are present in `degree_df`. If a gene in
        `unique_genes` is not found in `degree_df`, it will not be included in the returned DataFrame.
    """
    df = degree_df.copy()
    df = df.loc[unique_genes]
    return df

def _create_random_degree_matched_set(unique_gene_sets:dict, string_net_all_genes:list, string_net_degree_df:pd.DataFrame, seed:int) -> dict:
    """
    Generates random sets of genes matched by degree distribution to given gene sets.

    This function creates random gene sets that are degree-matched to the provided gene sets. For each gene set in
    `unique_gene_sets`, it generates a random set of genes from `string_net_all_genes` such that the degree distribution
    matches that of the original set. The degree information is sourced from `string_net_degree_df`.

    Args:
        unique_gene_sets (dict): A dictionary where keys are set identifiers and values are lists of genes in each set.
        string_net_all_genes (list): A list of all genes present in the STRING network.
        string_net_degree_df (pd.DataFrame): A DataFrame containing degree information for genes in the STRING network.
        seed (int): A seed for the random number generator to ensure reproducibility.

    Returns:
        dict: A dictionary where each key corresponds to a key in `unique_gene_sets`, and each value is a list of randomly
              selected genes. The random genes are selected such that their degree distribution matches that of the genes in
              the corresponding original set.

    Note:
        The function uses a numpy random number generator with the provided seed for reproducibility. It ensures that the
        randomly generated sets have a similar degree distribution as the original sets, which is crucial for certain types
        of network analysis where degree distribution is a significant factor.
    """
    random_sets = {}
    seed = seed *2
    rng = np.random.default_rng(seed)
    for k, v in unique_gene_sets.items():
        unique_mapped_genes = v
        # need to filter for genes that are mapped to STRING appropriately
        unique_mapped_genes = [
            x for x in unique_mapped_genes if x in string_net_all_genes]
        unique_mapped_genes_degree_df = _get_node_degree_dict(
            unique_mapped_genes, string_net_degree_df)
        unique_mapped_genes_degree_df = pd.DataFrame(
            unique_mapped_genes_degree_df.groupby('degree_rounded')['degree'].count())
        #in this dictionary: key is degree, value is count of genes with that degree
        unique_mapped_genes_degree_dict = dict(zip(unique_mapped_genes_degree_df.index.tolist(
        ), unique_mapped_genes_degree_df['degree'].tolist()))

        random_genes = []
        for k1, v1 in unique_mapped_genes_degree_dict.items():
            degree_df = string_net_degree_df.copy()
            degree_df = degree_df[degree_df['degree_rounded'] == k1]

            degree_matched_genes = degree_df.index.tolist()
            x = rng.choice(degree_matched_genes, v1, replace = False).tolist()
            random_genes.extend(x)
        random_sets[k] = random_genes
    return random_sets

def _plot_results_norm(random_dist:list, true_connections:list, fontface: str = 'Avenir', fontsize: int = 14, random_color: str = 'gray', true_color: str = 'red') -> Image:
    """
        Plots the normalized distribution of random distances and marks the average and true connection count.
    The plot is returned as a PIL Image object.

    Parameters
    ----------
    random_dist : List[float]
        A list of numerical values representing the random distribution.
    true_connections : List[str]
        A list of strings representing the true connections.
    fontface : str, optional
        The font face to use for all text in the plot, by default 'Avenir'.
    random_color : str, optional
        The color for the random distribution elements in the plot, by default 'gray'.
    true_color : str, optional
        The color for the true connections line in the plot, by default 'red'.

    Returns
    -------
    Image
        A PIL Image object of the plot.

    Examples
    --------
    >>> image = _plot_results_norm([0.5, 0.6, 0.7], ['conn1', 'conn2'])
    >>> image.show()  # This will display the image.
    >>> image.save('my_plot.png')  # This will save the image to a file.
    """
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _, ax = plt.subplots(tight_layout=True)
    ax.hist(random_dist, color=random_color, density=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.axvline(np.mean(random_dist), linestyle='dashed', color=random_color)
    plt.axvline(len(true_connections), linestyle='dashed', color=true_color)
    plt.xlabel('Number Intermethod Edges', size=14)
    plt.ylabel('Random Analysis Count', size=14)

    mu, sigma = norm.fit(random_dist)
    z_fit = (len(true_connections) - mu)/sigma
    xmin, xmax = plt.xlim()
    plt.xlim(xmin, xmax)
    plt.text(xmin + 5, 0.002, 'Z-score: ' + str(round(z_fit, 2)) + '\nNumber True Connections: ' + str(len(true_connections)),
             bbox={'facecolor': 'blue', 'alpha': 1.0, 'pad': 2}, color='white')

    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    ax.plot(x, p, linewidth=2)
    # Use PIL to save the plot
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    #buffer.close()
    plt.close()

    return image

def _output_unique_gene_network(unique_gene_network:pd.DataFrame, unique_genes:list) -> None:
    """
    Processes a gene network DataFrame to identify and save unique gene interactions.

    This function maps a list of unique genes to their sources in the network,
    filters out self-interactions, and saves the resulting network to a CSV file.

    Parameters
    ----------
    unique_gene_network : pd.DataFrame
        A DataFrame containing the gene network data with at least 'node1' and 'node2' columns.
    unique_genes : list
        A list mapping indices to gene names to identify the source of nodes.
    savepath : str
        The file path where the resulting CSV file will be saved.

    Returns
    -------
    None
        This function does not return a value; it writes output to a file.

    Examples
    --------
    >>> network_df = pd.DataFrame({
    ...     'node1': ['geneA', 'geneB', 'geneC'],
    ...     'node2': ['geneD', 'geneE', 'geneA'],
    ...     'score': [0.9, 0.8, 0.85]})
    >>> unique_genes = {'geneA': 'source1', 'geneB': 'source2', 'geneC': 'source3', 'geneD': 'source4', 'geneE': 'source5'}
    >>> _output_unique_gene_network(network_df, unique_genes, './')
    # This will save 'UniqueGeneNetwork.csv' to the current directory.
    """
    unique_gene_network['node1_source'] = unique_gene_network['node1'].map(
        unique_genes)
    unique_gene_network['node2_source'] = unique_gene_network['node2'].map(
        unique_genes)
    unique_gene_network = unique_gene_network[unique_gene_network['node1_source']
                                          != unique_gene_network['node2_source']]
    unique_gene_network = unique_gene_network[[
        'node1', 'node2', 'score', 'node1_source', 'node2_source']]
    return unique_gene_network

def _gene_set_overlap(gene_sets:dict) -> Image:
    """
    Generates and exports a Venn diagram visualizing the overlap between gene sets.

    This function takes a dictionary of gene sets, converts lists to sets for unique representation,
    generates a Venn diagram to visualize the overlaps, and saves the figure to a specified path.

    Parameters
    ----------
    gene_sets : dict
        A dictionary where each key is the name of the gene set and each value is a list of genes.
    savepath : str
        The file path where the Venn diagram figure will be saved, including the file name and extension.

    Returns
    -------
    None
        This function does not return a value; it exports a figure to the given save path.

    Examples
    --------
    >>> gene_sets = {
    ...     'Set1': ['gene1', 'gene2', 'gene3'],
    ...     'Set2': ['gene2', 'gene3', 'gene4'],
    ...     'Set3': ['gene1', 'gene4', 'gene5']
    ... }
    >>> _gene_set_overlap(gene_sets, 'gene_set_overlap.png')
    # This will save the Venn diagram as 'gene_set_overlap.png'.
    """
    gene_sets_ = {}
    for k, v in gene_sets.items():
        gene_sets_[k] = set(v)
    venn(gene_sets_, fontsize=8, legend_loc="upper left")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image

def _process_random_set(iteration:int, unique_gene_sets:dict, string_net_all_genes:pd.DataFrame, string_net_degree_df:pd.DataFrame, string_net:pd.DataFrame) -> Any:
    """
    Processes a single iteration for creating and analyzing a random degree-matched gene set.

    This function performs a series of operations to create a random degree-matched gene set based on a given iteration
    index, and then analyzes this set to determine the number of connections in the network. It encapsulates the workflow
    of generating random sets, mapping genes, and evaluating network connections.

    Args:
        iteration (int): The iteration index used as a seed for randomization.
        unique_gene_sets (dict): A dictionary of unique gene sets.
        string_net_all_genes (list): A list of all genes in the STRING network.
        string_net_degree_df (pd.DataFrame): A DataFrame containing degree information for genes in the STRING network.
        string_net (pd.DataFrame): A DataFrame representing the STRING network.

    Returns:
        int or None: The number of connections in the random unique gene network. Returns None if an error occurs.

    Note:
        This function is designed to be used in parallelized execution for multiple iterations of random set generation
        and analysis. It handles exceptions internally and prints error messages without halting execution.
    """
    try:
        random_sets = _create_random_degree_matched_set(unique_gene_sets, string_net_all_genes, string_net_degree_df, iteration)
        random_gene_sources = _get_gene_sources(random_sets)
        random_unique_genes = _get_unique_genes(random_gene_sources)
        random_unique_gene_network = _get_unique_gene_network(list(random_unique_genes.keys()), string_net)
        random_connections = _get_unique_gene_network_bw_method_connections(random_unique_gene_network, random_unique_genes)
        return len(random_connections)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None

def _parallel_random_enrichment(unique_gene_sets:dict, string_net_all_genes:pd.DataFrame, string_net_degree_df:pd.DataFrame, string_net:pd.DataFrame, num_iterations:int, num_processes:int) -> list:
    """
    Executes random enrichment analysis in parallel over multiple iterations.

    This function orchestrates the parallel execution of random set generation and network analysis across multiple iterations.
    It uses multiprocessing to distribute the task across multiple processes, thereby speeding up the computation significantly.

    Args:
        unique_gene_sets (dict): A dictionary of unique gene sets.
        string_net_all_genes (list): A list of all genes in the STRING network.
        string_net_degree_df (pd.DataFrame): A DataFrame containing degree information for genes in the STRING network.
        string_net (pd.DataFrame): A DataFrame representing the STRING network.
        num_iterations (int): The number of iterations to perform the random enrichment analysis.
        num_processes (int): The number of processes to use in parallel execution.

    Returns:
        list: A list of integers, each representing the number of connections in the random unique gene network for each iteration.

    Note:
        It is crucial to use this function within a 'if __name__ == "__main__":' block for safe parallel execution, especially
        on Windows. The number of processes (`num_processes`) should be chosen based on the system's capabilities.
    """
    args = [(i, unique_gene_sets, string_net_all_genes, string_net_degree_df, string_net) for i in range(num_iterations)]

    with Pool(num_processes) as pool:
        random_sets_connections = pool.starmap(_process_random_set, args)

    return random_sets_connections

def interconnectivity(set_1:list, set_2:list, set_3:list = None, set_4:list = None, set_5:list = None, savepath:Any = False, evidences:list = ['all'], edge_confidence:str = 'highest', num_iterations: int = 250, cores: int = 1, plot_fontface:str = 'Avenir', plot_fontsize:int = 14, plot_background_color:str = 'gray', plot_query_color: str = 'red') -> (Image, Image, list, pd.DataFrame, dict):
    """
    Analyzes gene set interconnectivity and visualizes the results, returning multiple outputs
    including images, lists, and data structures.

    Args:
        set_1 (list): Strings of gene names to network embed.
        set_2 (list): Strings of gene names to network embed.
        set_3 (list, optional): Strings of gene names to network embed. Defaults to None.
        set_4 (list, optional): Strings of gene names to network embed. Defaults to None.
        set_5 (list, optional): Strings of gene names to network embed. Defaults to None.
        savepath (str, optional): Path to the save directory. Defaults to './'.
        evidences (list, optional): Evidences list to calculate interaction score. Defaults to ['all'].
        edge_confidence (str, optional): Level of interactions to include. Defaults to 'highest'.
        num_iterations (int, optional): Number of random samplings from STRING network. Defaults to 250
        cores (int, optional): Number of cores for parallel jobs. Defaults to 1.
        plot_background_color (str, optional): Color of the background distribution. Defaults to 'gray'.
        plot_query_color (str, optional): Color of the query line. Defaults to 'red'.
        plot_fontface (str, optional): Font face for plot text. Defaults to 'Avenir'.
        plot_fontsize (int, optional): Font size for plot text. Defaults to 12.

    Returns:
        Image: A PIL Image object of the interconnectivity enrichment plot.
        Image: A PIL Image object of the Venn diagram illustrating gene set overlaps.
        list: A list of the number of connections in randomly generated gene sets.
        pd.DataFrame: A DataFrame detailing the unique gene network.
        dict: A dictionary of gene sources.

    Raises:
        ValueError: If the gene sets provided do not fit the expected input structure.

    Examples:
        >>> enrich_plot, venn_plot, random_set_connections, true_connection_df, query_gene_sources = _interconnectivity_enrichment(['geneA', 'geneB'], ['geneC', 'geneD'])
        This will perform the interconnectivity analysis and return the results.
        >>> resized_image = enrich_plot.resize((3, 4))
        Display the new image

    Note:
        If `savepath` is provided and not 'None', the function will save the results to disk
        at the specified path.
    """
    #load and customize STRINGv11 network for analysis (evidence types, edge weight)
    string_net = _load_string()
    string_net_all_genes = list(set(string_net['node1'].unique().tolist() + string_net['node2'].unique().tolist()))
    evidence_lst = _get_evidence_types(evidences)
    string_net = _select_evidences(evidence_lst, string_net)
    string_net = _get_combined_score(string_net)

    #Filtering network for edge weight
    edge_weight = _get_edge_weight(edge_confidence)
    string_net = string_net[string_net['score'] >= edge_weight]

    #get degree connectivity after edgeweight filtering
    # network is already edge weight filtered
    g_string_net = nx.from_pandas_edgelist(
        string_net[['node1', 'node2']], 'node1', 'node2')
    g_string_net_degree = dict(g_string_net.degree)
    string_net_degree_df = pd.DataFrame(index=string_net_all_genes)
    string_net_degree_df['degree'] = string_net_degree_df.index.map(
        g_string_net_degree)
    # fillna with zeros for genes w/o appropriate edge weights
    string_net_degree_df.fillna(0, inplace=True)
    # Round each degree to nearest 10
    string_net_degree_df['degree_rounded'] = string_net_degree_df['degree'].apply(
        lambda x: round(x/10)*10)
    #true gene sets b/w set unique gene connectivity
    gene_sets = _load_genelists(set_1, set_2, set_3, set_4, set_5)
    query_gene_sources = _get_gene_sources(gene_sets)
    query_unique_genes = _get_unique_genes(query_gene_sources)
    query_unique_gene_network = _get_unique_gene_network(
        list(query_unique_genes.keys()), string_net)
    true_connections = _get_unique_gene_network_bw_method_connections(query_unique_gene_network, query_unique_genes)
    # random gene sets b/w set unique gene connectivity w/ degree matching
    # dictionary with unique genes (values) per each set (keys) in true gene lists
    unique_gene_sets = _get_unique_gene_counts(query_unique_genes)
    # Perform random enrichment-parallelized
    random_sets_connections = _parallel_random_enrichment(unique_gene_sets, string_net_all_genes, string_net_degree_df, string_net, num_iterations, cores)
    # Generate z score
    mu, sigma = norm.fit(random_sets_connections)
    z_fit = (len(true_connections) - mu)/sigma
    # Plot results
    enrich_image = _plot_results_norm(random_sets_connections, true_connections, fontface = plot_fontface, fontsize = plot_fontsize, random_color = plot_background_color, true_color = plot_query_color)
    # Get output files
    venn_image = _gene_set_overlap(gene_sets)
    unique_gene_network = _output_unique_gene_network(query_unique_gene_network, query_unique_genes)

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'GoldStandard_Interconnectivity/')
        os.makedirs(new_savepath, exist_ok=True)
        enrich_image.save(new_savepath + "Interconnectivity_Norm_Plot.png", format = "PNG")
        venn_image.save(new_savepath + "Interconnectivity_Venn.png", format = "PNG")
        pd.DataFrame({'col': random_sets_connections}).to_csv(new_savepath + "RandomSetConnections.csv", header = False, index = False)
        unique_gene_network.to_csv(new_savepath + "UniqueGeneNetwork.csv", header = True, index = False)
        pd.DataFrame.from_dict(query_gene_sources, orient='index').to_csv(new_savepath + "GeneSet_Sources.csv", header = False)
        # Write stats
        with open(new_savepath + "EnrichmentStats.txt", "w") as f:
            f.write('Number of b/w set connects in real gene sets:'+ str(len(true_connections)) + '\n')
            f.write('Z-score based on curve fit:' + str(z_fit))
        f.close()
    print('Number of b/w set connects in real gene sets:', len(true_connections))
    print('Z-score based on curve fit:', z_fit)
    return venn_image, enrich_image, random_sets_connections, unique_gene_network, query_gene_sources
#endregion



#region GWAS Colocalization
def _map_mondo_efo_id(mondo_id: str) -> Any:
    """
    Maps a MONDO ID to its corresponding EFO (Experimental Factor Ontology) ID using the OLS (Ontology Lookup Service) API.

    This function queries the OLS API to find the exact mapping of a given MONDO(Mendelian Inheritance in Man Ontology) ID to an EFO ID.
    It returns the first matching EFO ID found.

    Args:
        mondo_id (str): The MONDO ID to be mapped to an EFO ID. The MONDO ID should be in the format 'MONDO:XXXXXXX'.

    Returns:
        str: The corresponding EFO ID in the format 'EFO_XXXXXXX'. Returns False if no matching EFO ID is found.

    Raises:
        HTTPError: If the HTTP request to the OLS API fails.
        JSONDecodeError: If the response from the OLS API is not in JSON format or is otherwise malformed.

    Note:
        The function assumes that the MONDO ID is correctly formatted and that the OLS API endpoint is accessible and functioning.
        It retrieves the first exact match found and does not account for multiple possible mappings.
    """
    # Define the OLS API endpoint and parameters
    ols_api_url = "https://www.ebi.ac.uk/ols/api/terms"
    params = {
        "obo_id": mondo_id,
        "ontology": "efo",
        "mapping_types": "exact",
    }

    # Query the OLS API
    response = requests.get(ols_api_url, params=params)
    response.raise_for_status()
    data = response.json()

    # Check if data is available
    if not data.get("_embedded"):
        return False
    else:
        efo_term = data["_embedded"]["terms"][0]["obo_id"]
        efo_term = efo_term.replace(":", "_")
        return efo_term

def _pull_gwas_catalog(mondo_id:str, p_upper:float) -> pd.DataFrame:
    """
    Pulls GWAS (Genome-Wide Association Studies) catalog data for a given MONDO ID and filters the results based on a p-value threshold.

    This function first checks if the input MONDO ID is in the correct format, and if not, attempts to map it to the correct EFO (Experimental Factor Ontology) ID using the _map_mondo_efo_id function. It then queries the GWAS Catalog API to download association data for the given MONDO/EFO ID. The data is filtered to include only records with a p-value less than or equal to the specified upper limit (p_upper).

    Args:
        mondo_id (str): The MONDO ID for which GWAS data is to be pulled. It can be in MONDO format or a format that can be mapped to EFO.
        p_upper (float): The upper limit for p-values to include in the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing GWAS catalog data filtered by the specified p-value threshold.

    Raises:
        ValueError: If no corresponding EFO term is found for the input MONDO ID, or if the GWAS Catalog data cannot be downloaded.

    Notes:
        The function requires an internet connection to access the GWAS Catalog API. The API URL is hardcoded for 'associations/download' endpoint with 'includeBgTraits' and 'includeChildTraits' set to True.
    """
    if "MONDO_" not in mondo_id:
        mondo_id = _map_mondo_efo_id(mondo_id)
    if not mondo_id:
        raise ValueError(f"No EFO term found for {mondo_id}, please download and add the path to the table.")

    print(f"Querying GWAS Catalog API for {mondo_id}")
    api_url = f"https://www.ebi.ac.uk/gwas/api/v2/efotraits/{mondo_id}/associations/download?includeBgTraits=True&includeChildTraits=True"
    response = requests.get(api_url)

    if response.status_code == 200:
        content_file = io.BytesIO(response.content)
        df = pd.read_csv(content_file, sep = '\t')
        df = df[df.pValue <= p_upper]
        df_temp = df['locations'].str.split(":", expand = True)
        df['CHR_ID'] = df_temp[0]
        df['CHR_POS'] = df_temp[1]
        df.drop(columns = ['locations'], inplace = True)
        download_columns = ['STRONGEST SNP-RISK ALLELE', 'P-VALUE', 'P-VALUE (TEXT)', 'RISK ALLELE FREQUENCY', 'OR VALUE', 'BETA', '95% CI (TEXT)', 'MAPPED_GENE', 'DISEASE/TRAIT','MAPPED_TRAIT', 'BG_TRAITS', 'STUDY ACCESSION', 'PUBMEDID', 'FIRST AUTHOR', 'CHR_ID', 'CHR_POS',]
        df.columns = download_columns
        df.set_index('STRONGEST SNP-RISK ALLELE', inplace = True)
        return df, mondo_id
    else:
        print(f"Status code exited with error: {response.status_code}")
        raise ValueError(f"Cannot download summary statistics for {mondo_id}, please download and add the path to the table.")

def _get_gene_loci(g:str, geneloci:pd.DataFrame) -> (int, int, int):
    """
    Retrieves the chromosome location (chromosome number, start position, and end position) for a specified gene.

    This function looks up a given gene in a DataFrame containing gene loci information and returns the chromosome number,
    start position, and end position of the gene.

    Args:
        g (str): The gene symbol for which loci information is to be retrieved.
        geneloci (pd.DataFrame): A pandas DataFrame containing gene loci information.
                                 This DataFrame must have columns 'chrom', 'start', and 'end'.
    Returns:
        tuple:
            - int: The chromosome number on which the gene is located.
            - int: The start position of the gene on the chromosome.
            - int: The end position of the gene on the chromosome.
    Raises:
        KeyError: If the specified gene is not found in the DataFrame.
        ValueError: If the start or end positions cannot be converted to integers.

    Note:
        The function assumes that the 'chrom' column in the DataFrame contains chromosome numbers as integers.
    """
    row = geneloci.loc[g].values
    g_chr, g_start, g_end = row[0], row[1], row[2]
    return g_chr, int(g_start), int(g_end)

def _get_snp_loci(snp:str, gwasloci:pd.DataFrame) -> (int, int):
    """
    Retrieves the chromosome location (chromosome number and position) for a specified SNP (Single Nucleotide Polymorphism).

    This function looks up a given SNP in a DataFrame containing GWAS (Genome-Wide Association Studies) loci information
    and returns the chromosome number and position of the SNP.

    Args:
        snp (str): The SNP identifier for which loci information is to be retrieved.
        gwasloci (pd.DataFrame): A pandas DataFrame containing GWAS loci information.
                                 This DataFrame must have columns 'CHR_ID' and 'CHR_POS'.

    Returns:
        tuple:
            - int: The chromosome number on which the SNP is located, or None if not found/convertible.
            - int: The position of the SNP on the chromosome, or None if not found/convertible.

    Raises:
        KeyError: If the specified SNP is not found in the DataFrame.

    Note:
        The function attempts to convert the chromosome position to an integer. If this conversion fails,
        it returns None for both chromosome number and position. The function assumes that 'CHR_ID' column
        in the DataFrame contains chromosome numbers and 'CHR_POS' contains position numbers.
    """
    s_chr = gwasloci.loc[snp]['CHR_ID']
    s_pos = gwasloci.loc[snp]['CHR_POS']
    try:
        return s_chr, int(s_pos)
    except ValueError:
        return None, None

def _get_snp_distance(snp:int, g_loc:int) -> int:
    """
    Calculates the absolute distance between a SNP position and a gene location.

    This function computes the absolute difference between the position of a Single Nucleotide
    Polymorphism (SNP) and a gene location on a chromosome. This is useful for understanding
    the physical distance between a SNP and a gene, which can be relevant in genetic studies.

    Args:
        snp (int): The position of the SNP on the chromosome.
        g_loc (int): The position of the gene on the chromosome.

    Returns:
        int: The absolute distance between the SNP and the gene location.

    Note:
        Both the SNP position and gene location should be provided as integer values representing
        their respective positions on a chromosome. The function uses numpy's absolute value function
        to ensure the distance is returned as a positive integer.
    """
    return int(np.abs(snp - g_loc))

def _find_snps_within_range(query_genes:list, ref_gene_index:list, geneloci:pd.DataFrame, gwasloci:pd.DataFrame, query_distance:int) -> dict:
    """
    Identifies Single Nucleotide Polymorphisms (SNPs) within a specified distance from given query genes.

    This function iterates through a list of query genes and identifies SNPs from a GWAS (Genome-Wide Association Studies) loci
    DataFrame that are within a specified distance from each gene. It returns adictionary where each query gene is mapped to a list
    of SNPs (and their distances) that are within the defined range of that gene.
    Args:
        query_genes (list): A list of gene symbols to query.
        ref_gene_index (list): A list of SNP identifiers to reference against.
        geneloci (pd.DataFrame): A pandas DataFrame containing gene loci information with columns 'chrom', 'start', and 'end'.
        gwasloci (pd.DataFrame): A pandas DataFrame containing GWAS loci information with columns 'CHR_ID' and 'CHR_POS'.
        query_distance (int): The distance within which to search for SNPs near each query gene, in base pairs.
    Returns:
        dict: A dictionary mapping each query gene to a list of SNPs (and distances) that are within the specified distance of the gene.
    Note:
        The function checks for chromosome number compatibility (autosomes or sex chromosomes) and calculates distances in kilobase pairs (Kbp).
        It handles exceptions for genes or SNPs not found in the provided DataFrames and skips to the next gene/SNP if an issue is encountered.
    """
    # Get gene location information
    gene_loci = geneloci.loc[query_genes, ['chrom', 'start', 'end']]
    # Get SNP location information
    snp_loci = gwasloci.loc[ref_gene_index, ['CHR_ID', 'CHR_POS', 'MAPPED_GENE']]
    # Initialize the output dictionary
    gene_dict = {}
    # Iterate through genes and SNPs, and calculate the distance between them
    for gene, gene_data in gene_loci.iterrows():
        snps_within_range = []
        for snp, snp_data in snp_loci.iterrows():
            if gene_data['chrom'] == snp_data['CHR_ID']:
                try:
                    dist1 = abs(int(gene_data['start']) - int(snp_data['CHR_POS']))
                    dist2 = abs(int(gene_data['end']) - int(snp_data['CHR_POS']))
                except ValueError:
                    continue
                if dist1 < query_distance or dist2 < query_distance:
                    min_dist = min(dist1, dist2)
                    snps_within_range.append(
                        f"{snp} ({snp_data['MAPPED_GENE']}; {round(min_dist / 1000, 1)} Kbp)")
        gene_dict[gene] = snps_within_range

    return gene_dict

def _count_genes_with_snps(gene_snp_dict:dict) -> int:
    """
    Counts the number of genes that have associated SNPs within a given distance.

    This function takes a dictionary mapping genes to SNPs and returns the count of genes that have one or more SNPs associated with them.
    It is specifically designed to work with the output of a function that maps genes to nearby SNPs.

    Args:
        gene_snp_dict (dict): A dictionary where keys are gene symbols and values are lists of SNPs.
                              Each SNP is associated with the gene as a nearby genetic variant.

    Returns:
        int: The number of genes that have at least one SNP associated with them.

    Note:
        The function does not count the total number of SNPs but rather counts the number of genes that have SNPs.
        A gene with multiple SNPs is counted as one.
    """
    count = sum(1 for snps in gene_snp_dict.values() if snps)
    return count

def _get_genes_with_snps(gene_snp_dict:dict) -> list:
    """
    Retrieves a list of genes that have associated SNPs within a specified distance.

    This function processes a dictionary where genes are mapped to SNPs and returns a list of those genes that have
    one or more SNPs associated with them. It is particularly useful for identifying genes with nearby genetic variants.

    Args:
        gene_snp_dict (dict): A dictionary where keys are gene symbols and values are lists of SNPs.
                              Each SNP is considered to be associated with the gene as a nearby genetic variant.

    Returns:
        list: A list of genes that have at least one associated SNP.

    Note:
        The function only includes genes that have associated SNPs. Genes without any associated SNPs are excluded from the returned list.
    """
    genes = [gene for gene, snps in gene_snp_dict.items() if snps]
    return genes

def _run_parallel_query(function:callable, chunks:list, gwas_snp_ids:pd.Series, geneloci:pd.DataFrame, gwasloci:pd.DataFrame, distance:int, num_processes:int) -> list:
    """
    Executes a given function in parallel across multiple processes.

    This function takes a user-defined function and applies it to each chunk of data in parallel using Python's
    multiprocessing capabilities. It is designed to perform computationally intensive tasks more efficiently by
    distributing the workload across multiple CPU cores.

    Args:
        function (callable): The function to be executed in parallel. This function should accept the parameters
                             as defined in the tuple structure used in the starmap call.
        chunks (list): A list of data chunks on which the function will be executed. Each chunk is processed
                       independently in a separate process.
        gwas_snp_ids (list): A list of SNP identifiers used in the function.
        geneloci (pd.DataFrame): A pandas DataFrame containing gene loci information.
        gwasloci (pd.DataFrame): A pandas DataFrame containing GWAS loci information.
        distance (int or float): The distance parameter to be passed to the function.
        num_processes (int): The number of parallel processes to use.

    Returns:
        list: A list of results obtained from executing the function on each chunk of data.

    Note:
        The user-defined function passed to this function must be capable of handling the data structure
        contained in each chunk and must accept the same parameters as those passed to this function.
    """
    with Pool(num_processes) as pool:
        results = pool.starmap(function, [(chunk, gwas_snp_ids, geneloci, gwasloci, distance) for chunk in chunks])
        pool.close()
        pool.join()
    return results

def _chunk_data(data:list, num_chunks:int) -> list:
    """
    Splits a list of data into smaller, approximately equal-sized chunks.

    This function divides a given list into a specified number of chunks. It is primarily used to partition data for
    parallel processing, ensuring that each chunk is as evenly distributed as possible.

    Args:
        data (list): The list of data to be chunked. This can be a list of any type of elements.
        num_chunks (int): The number of chunks to divide the data into. It should be a positive integer.

    Returns:
        list: A list of chunks, where each chunk is a sublist of the original data. The size of each chunk is approximately
              equal to the total size of the data divided by `num_chunks`.

    Note:
        If the total size of the data is not evenly divisible by `num_chunks`, the last chunk may be smaller than the others.
        This function does not shuffle or alter the order of the original data elements.
    """
    chunk_size = len(data) // num_chunks + 1
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def _combine_dicts(dicts_list:list) -> dict:
    """
    Merges a list of dictionaries into a single dictionary.

    This function takes a list of dictionaries and combines them into one dictionary. In the resulting dictionary,
    each key-value pair from each dictionary in the input list is preserved. If the same key appears in multiple
    dictionaries, the value from the last dictionary with that key will be used in the merged dictionary.

    Args:
        dicts_list (list): A list of dictionaries to be merged. Each element in the list should be a dictionary.

    Returns:
        dict: A single dictionary that contains the combined key-value pairs from all dictionaries in the input list.

    Note:
        If there are duplicate keys across the dictionaries, the value associated with the key in the last dictionary
        containing that key will overwrite any previous value. The order of dictionaries in the list impacts the final content.
    """
    return {k: v for d in dicts_list for k, v in d.items()}

def _calculate_fishers_exact(gene_snp_dict:dict, all_gene_dict:dict, final_genes:list) -> (int, int, int, int, float):
    """
    Calculates Fisher's Exact Test for a set of genes and their associated SNPs.

    This function computes the Fisher's Exact Test on a 2x2 contingency table formed from gene and SNP associations.
    It calculates the counts of true positives, false positives, false negatives, and true negatives based on the provided dictionaries.

    Args:
        gene_snp_dict (dict): A dictionary mapping genes to SNPs, used to determine the counts of true and false positives.
        all_gene_dict (dict): A dictionary mapping all genes to their associated SNPs, used for the total count of gene-SNP associations.
        final_genes (list): A list of genes considered as true positives.

    Returns:
        tuple: A tuple containing the counts of true positives (tp), false positives (fp), false negatives (fn), true negatives (tn),
               and the p-value (pval) from Fisher's Exact Test. The structure is (tp, fp, fn, tn, pval).

    Note:
        The function assumes that 'final_genes' are the true positives. It calculates the true positives (tp) as the length of 'final_genes',
        false positives (fp) as the number of genes in 'gene_snp_dict' not in 'final_genes', false negatives (fn) as genes in
        'all_gene_dict' with SNPs but not in 'final_genes', and true negatives (tn) as genes not associated with any SNPs in both dictionaries.
        The p-value is calculated using the 'greater' alternative hypothesis in Fisher's Exact Test, indicating a one-tailed test.
    """
    tp = len(final_genes)
    fp = len(gene_snp_dict.keys()) - tp
    fn = len([k for k, v in all_gene_dict.items() if len(v) > 0]) - tp
    tn = len([x for x in all_gene_dict.keys() if x not in final_genes and x not in gene_snp_dict.keys()])
    _, pval = fisher_exact(np.array([[tp, fp], [fn, tn]]), alternative='greater')
    return tp, fp, fn, tn, pval

def gwas_catalog_colocalization(query:list, mondo_id:str = False, gwas_summary_path:str = False, gwas_p_thresh: float = 5e-8, distance_mbp:float = 0.5, cores:int = 1, savepath:Any = False, save_summary_statistics:bool = False) -> (pd.DataFrame, float):
    """
    Performs colocalization analysis between a list of query genes and GWAS catalog SNPs.

    This function identifies SNPs from a GWAS catalog that are within a specified physical distance from the query genes. It can
    optionally perform this analysis for a background gene set to assess the statistical significance of the colocalization using
    Fisher's Exact Test.

    Args:
        query (list): A list of query gene symbols.
        mondo_id (str, optional): The MONDO ID to pull GWAS catalog data. If provided, GWAS catalog data will be pulled based on this ID.
        gwas_summary_path (str, optional): The file path to a pre-downloaded GWAS summary statistics file. Used if 'mondo_id' is not provided.
        gwas_p_thresh (float, optional): The p-value threshold for filtering GWAS catalog SNPs. Defaults to 5e-8.
        distance_mbp (float, optional): The distance in megabase pairs (Mbp) within which to search for SNPs around each gene. Defaults to 0.5 Mbp.
        cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        savepath (Any, optional): The path to save output files. If False, no files are saved.
        save_summary_statistics (bool, optional): Flag to save GWAS summary statistics. Effective only if 'savepath' is provided.

    Returns:
        pd.DataFrame: A DataFrame with genes and their associated SNPs within the specified distance.
        float: The p-value from Fisher's Exact Test assessing the significance of the colocalization.

    Note:
        The function uses multiprocessing for parallel computation if 'cores' is greater than 1. Ensure that the function is called
        within a 'if __name__ == "__main__":' block for safe parallel execution, especially on Windows platforms.
    """
    # Load GWAS Summary Stats
    if mondo_id:
        gwas_catalog, mondo_id = _pull_gwas_catalog(mondo_id, gwas_p_thresh)
    elif gwas_summary_path:
        gwas_catalog = pd.read_csv(gwas_summary_path, sep = '\t')
        gwas_catalog = gwas_catalog[gwas_catalog['P-VALUE'] <= gwas_p_thresh]
        gwas_catalog.set_index('STRONGEST SNP-RISK ALLELE', inplace = True)
        mondo_id = ""
    # Set parameters for colocalization
    distance_bp = distance_mbp * 1000000
    gene_locations = _load_grch38_background(just_genes=False)
    gwas_catalog = gwas_catalog[['CHR_ID', 'CHR_POS', 'MAPPED_GENE']].drop_duplicates()
    # Run colocalization for query
    print("Running query genes")
    query_chunks = _chunk_data(query, cores)
    query_gene_dicts = _run_parallel_query(_find_snps_within_range, query_chunks, gwas_catalog.index, gene_locations, gwas_catalog, distance_bp, cores)
    query_snp_dict = _combine_dicts(query_gene_dicts)
    query_snp_df = pd.DataFrame(query_snp_dict.items(), columns=['Gene', 'SNPs'])
    final_genes = _get_genes_with_snps(query_snp_dict)
    # Run colocalization for background
    print("Running background genes")
    bg_chunks = _chunk_data(gene_locations.index.tolist(), cores)
    bg_gene_dicts = _run_parallel_query(_find_snps_within_range, bg_chunks, gwas_catalog.index, gene_locations, gwas_catalog[['CHR_ID', 'CHR_POS', 'MAPPED_GENE']].drop_duplicates(), distance_bp, cores)
    bg_gene_dict = _combine_dicts(bg_gene_dicts)
    # Test significance
    tp, fp, fn, tn, pval = _calculate_fishers_exact(query_snp_dict, bg_gene_dict, final_genes)
    # Output values
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'GWAS_Colocalization/')
        os.makedirs(new_savepath, exist_ok=True)
        if save_summary_statistics:
            gwas_catalog.to_csv(new_savepath + f"GWAS_Colocalization_{mondo_id}_p-{gwas_p_thresh}.csv", index = True)
        query_snp_df.to_csv(new_savepath +f"GWAS_Colocalization_{mondo_id}_p-{gwas_p_thresh}_TP.csv", index = False)
        with open(new_savepath + f"GWAS_Colocalization_{mondo_id}_p-{gwas_p_thresh}_Summary.txt", "w") as f:
            f.write(f"TP = {tp}\n")
            f.write(f"FP = {fp}\n")
            f.write(f"FN = {fn}\n")
            f.write(f"TN = {tn}\n")
            f.write(f"P-value = {pval}")
            f.close()

    return query_snp_df, pval
#endregion



#region PubMed Co-mentions
def _parse_field(field: str) -> Any:
    """
    Parses the input field and returns the appropriate query field.

    Args:
    ----
    field (str): The field specified by the user.

    Returns:
    -------
    str: The processed field value for the query.

    Raises:
    ------
    ValueError: If the provided field is not a valid selection.
    """
    valid_fields = ['all', 'title/abstract', 'title']
    if field not in valid_fields:
        raise ValueError(f"Invalid field selection '{field}'. Select a permitted field type: {', '.join(valid_fields)}")
    if field == 'all':
        return None
    return field

def _entrez_search(gene:str, disease:str, email:str, api_key:str, field:str) -> 'Entrez.Parser.DictionaryElement':
    """
    Conducts a search on the Entrez (PubMed) database for articles related to a specific gene and disease.

    Utilizes the Biopython Entrez API to query the PubMed database for articles
    containing the specified gene and disease terms within the specified field.

    Args:
        gene (str): The gene of interest.
        disease (str): The disease of interest.
        field (str, optional): The field within the PubMed article to search. Defaults to "all".
        email (str): The email address associated with the Entrez account.
        api_key (str): The API key associated with the Entrez account.

    Returns:
        Bio.Entrez.Parser.DictionaryElement: A dictionary-like object containing the Entrez search results.

    Raises:
        ValueError: If the provided field is not a valid selection.
    """
    Entrez.email = email
    Entrez.api_key = api_key
    field = _parse_field(field)
    new_query = f'"{gene}" AND ("{disease}") AND ("gene" or "protein")'
    try: handle = Entrez.esearch(
        db = 'pubmed',
        sort = 'relevance',
        retmax = '100000',
        retmode = 'xml',
        field = field,
        term = new_query
    )
    except IndexError:
        print(f"{gene} - Not Found")
    except HTTPError:
        print('....Network Error-Waiting 10s')
        time.sleep(10)
        handle = Entrez.esearch(db = 'pubmed',
                            sort = 'relevance',
                            retmax = '100000',
                            retmode = 'xml',
                            term = new_query)
    except IncompleteRead:
        print('....Network Error-Waiting 10s')
        time.sleep(10)
        try: handle = Entrez.esearch(db = 'pubmed',
                            sort = 'relevance',
                            retmax = '100000',
                            retmode = 'xml',
                            term = new_query)
        except IncompleteRead:
            return "IncompleteReadError"

    results = Entrez.read(handle)
    handle.close()
    return results

def _parse_entrez_result(result:dict) -> (str, int):
    """
    Parses the result from an Entrez (PubMed) API query to extract gene information and the count of papers.

    This function attempts to extract the gene name and the number of papers associated with it from the result
    of a PubMed query. It handles different scenarios where the expected fields might not be directly available
    or are located in different parts of the result structure due to warnings or errors in the query.

    Args:
        result (dict): The result dictionary returned by an Entrez (PubMed) API query.

    Returns:
        tuple:
            - str: The extracted gene name from the query result. If the gene name cannot be found directly,
                   it attempts to extract it from the 'WarningList' or 'QueryTranslation'.
            - int: The number of papers associated with the gene. This is NaN if the count cannot be
                   determined due to a warning or an error in the query.

    Raises:
        KeyError: If the necessary keys are not found in the result dictionary.
        IndexError: If the extraction from 'QueryTranslation' fails due to index errors.

    Note:
        The function assumes that the 'result' dictionary follows the structure of a response from the
        Entrez (PubMed) API. It is designed to handle cases where the query might have warnings or errors.
    """
    try:
        gene = result['TranslationStack'][0]['Term'].split('"')[1]
        n_paper_dis = int(result['Count'])
    except KeyError:
        try:
            gene = result['WarningList']['QuotedPhraseNotFound'][0].split('"')[1]
            n_paper_dis = 0
        except KeyError:
            gene = result['QueryTranslation'].split('"')[1]
            n_paper_dis = int(result['Count'])
        except IndexError:
            gene = result['QueryTranslation'].split('"')[1]
            n_paper_dis = int(result['Count'])
    return gene, n_paper_dis

def _fetch_query_pubmed(query: list, keyword: str, email: str, api_key: str, field:str, cores: int) -> pd.DataFrame:
    """
    Fetches publication data from PubMed for a list of genes related to a specific keyword.
    This function concurrently queries PubMed for each gene and compiles the results into
    a DataFrame which includes the count of papers and the PMIDs of papers related to each gene.

    Args:
        query (List[str]): A list of genes to be queried.
        keyword (str): A keyword to be used in conjunction with genes for querying PubMed.
        email (str): The email address associated with the NCBI account for Entrez queries.
        api_key (str): The API key for making requests to the NCBI Entrez system.
        cores (int): The number of worker threads to use for concurrent requests.

    Returns:
        pd.DataFrame: A DataFrame indexed by gene names with the count of related papers and PMIDs.
    """
    # Initialize data frames to store the query results and output data
    out_df = pd.DataFrame(columns=['Count', 'PMID for Gene + ' + str(keyword)], index=query)

    # Check field validity
    if field not in ['all', 'title/abstract', 'title']:
        raise ValueError(f"Invalid field selection '{field}'. Select a permitted field type: all, title/abstract, title")

    # Execute concurrent API calls to PubMed
    with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
        results = executor.map(_entrez_search, query, repeat(keyword), repeat(email), repeat(api_key), repeat(field))
        # Process results for each gene
        for result in results:
            gene, n_paper_dis = _parse_entrez_result(result)
            # Populate the data frames with the results
            out_df.loc[gene, 'PMID for Gene + ' + keyword] = "; ".join(result.get('IdList', []))
            out_df.loc[gene, 'Count'] = n_paper_dis

    # Sort the output data frame by the count of related papers in descending order
    sorted_out_df = out_df.sort_values(by='Count', ascending=False)

    return sorted_out_df

def _fetch_random_pubmed(query: list, disease_query: str, email: str, api_key: str, cores: int, field:str, trials:int, background_genes:list) -> list:
    """
    Performs PubMed queries on random sets of genes and records the number of papers
    associated with a disease for each gene in the set.
    This function is used to establish a baseline or control for comparison with
    a set of query genes of interest.
    Args:
        query (List[str]): A list of genes to be used as a query size reference.
        background_genes (List[str]): A list of background genes from which random samples are drawn.
        disease_query (str): A disease term to be used in conjunction with genes for querying PubMed.
        email (str): The email address associated with the NCBI account for Entrez queries.
        api_key (str): The API key for making requests to the NCBI Entrez system.
        max_workers (int): The number of worker threads to use for concurrent requests.
        trials (int, optional): The number of random gene sets to query. Defaults to 100.
    Returns:
        List[pd.DataFrame]: A list of DataFrames, each containing the count of papers for a random gene set.
    """
    randfs = []
    if len(background_genes) == 0:
        background_genes = _load_grch38_background()
    print(f'Pulling Publications for {trials} random gene sets of {len(query)} genes')

    for i in range(trials):
        if i % 10 == 0:
            print(f" Random Trial : {i}")
        rng = np.random.default_rng(i*3)
        randgenes = rng.choice(background_genes, size = len(query), replace = False).tolist()
        tempdf = pd.DataFrame(columns=['Count'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=cores) as executor:
            # Map the search function over the random genes
            results = executor.map(_entrez_search, randgenes, repeat(disease_query), repeat(email), repeat(api_key), repeat(field))
            # Process the results and update the temporary DataFrame
            for result in results:
                gene, n_paper_dis = _parse_entrez_result(result)
                n_paper_dis = result.get('Count', 0)
                tempdf.loc[gene, 'Count'] = int(n_paper_dis)

        # Append the temporary DataFrame to the list
        randfs.append(tempdf)

    return randfs

def _calculate_z_score(observation: int, background: list) -> float:
    """
    Calculates the Z-score for the observed count against a background distribution.

    Args:
        observation (int): The observed count of genes.
        background (list): A list of counts from random gene sets for comparison.

    Returns:
        float: The calculated Z-score.
    """
    back_mean = np.mean(background)
    back_std = np.std(background)
    return (observation - back_mean) / back_std if back_std > 0 else np.nan

def _plot_results(disease_query: str, background: list, observation: int, query:list, paper_thrshld: list, z: float, random_color:str, query_color: str, fontsize: int, fontface:str) -> None:
    """
    Plots the results of the enrichment analysis.

    Args:
        disease_query (str): The disease query used in the enrichment analysis.
        background (list): A list of counts from random gene sets for comparison.
        observation (int): The observed count of genes.
        paper_thrshld (list): The threshold range for paper counts.
        z (float): The calculated Z-score.
        query (list): The list of query genes.
    """
    plt.rcParams.update({'font.size': fontsize,
                         'font.family': fontface})
    _, _ = plt.subplots(figsize=(6, 3.5), facecolor='white')
    y,_, _ = plt.hist(background, color=random_color)
    plt.axvline(x=observation, ymax=0.5, linestyle='dotted', color=query_color)
    # Orient the annotation
    values = background
    values.append(observation)
    if observation >= np.mean(values):
        ha_hold = 'right'
    else:
        ha_hold = 'left'
    plt.text(x=observation*0.99, y=(y.max()/1.8), s='{}/{} (Z = {})'.format(observation, len(query), round(z, 2)), color='red', ha=ha_hold)
    plt.xlabel('# of Genes with {}-{} Co-Mentions with "{}"'.format(paper_thrshld[0]+1, paper_thrshld[1], disease_query), fontsize=15)
    plt.ylabel('# of Random Occurrences', fontsize=15)
    plt.tight_layout()
    # Return image
    buffer = io.BytesIO()
    plt.savefig(buffer, format = 'png', dpi = 300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()

    return image

def pubmed_comentions(query:list, keyword: str, background_genes: list = [], field:str = 'all', email:str = 'kwilhelm95@gmail.com', api_key: str = '3a82b96dc21a79d573de046812f2e1187508', enrichment_trials: int = 100, workers: int = 15, run_enrichment:bool = True, enrichment_cutoffs:list = [[-1,0], [0,5], [5,15], [15,50], [50,100000]], plot_background_color:str = 'gray', plot_query_color: str = 'red', plot_fontface:str = 'Avenir', plot_fontsize:int = 14, savepath:Any = False) -> (pd.DataFrame, dict, dict):
    """
    Searches PubMed for comention of genes within articles related to a given field and
    performs a randomization test to compute Z-scores for observed mention counts.

    Args:
        query (list): A list of genes/other words to query in PubMed.
        keyword (str): A keyword so search with (i.e. "Type 2 Diabetes")
        field (str, optional): The field within the PubMed article to search. Defaults to "all".
        email (str, optional): The email address associated with the Entrez account.
        api_key (str, optional): The API key associated with the Entrez account.
        enrichment_trials (int, optional): The number of random trials to perform for calculating enrichment. Default is 100.
        cores (int, optional): Number of workers for querying PubMed. Default is 15.
        run_enrichment (bool, optional): False user only wants query co-mentions without enrichment. Default is True.
        enrichment_cutoffs (list, optional): Cutoffs for enrichment analyses. Logic = >1st number, <=2nd number. Default is [[-1,0], [0,5], [5,15], [15,50], [50,100000]].
        plot_background_color (str, optional): Color for random samplings in histogram. Default = gray.
        plot_query_color (str, optional): Color for query findings in histogram. Default = red.
        plot_fontface (str, optional): Fontface for plot. Default = Avenir.
        plot_fontsize (int, optional): Font size for plot. Default = 14.
        savepath (Any, optional): Path to save files. If undeclared, files are not saved and only returned.
    Returns:
        pd.DataFrame : Dataframe of query words, number of co-mentions, and PMIDs.
        dict : Dictionary, where keys = enrichment_cutoff values and values = (number of query genes in subset, z_score)
        dict : Dictionary, where keys = enrichment_cutoff values and values = enrichment plot.
    """
    # Pull the query co_mentions with keyword
    query_comention_df = _fetch_query_pubmed(query, keyword, email, api_key, field, workers)

    # Pull co_mentions for a random set of genes
    if run_enrichment:
        rand_dfs = _fetch_random_pubmed(query, keyword, email, api_key, workers, field, enrichment_trials, background_genes)
        enrich_results, enrich_images = {}, {}
        rand_result_df = pd.DataFrame({'Iteration': range(0, len(rand_dfs))})
        for min_thresh, max_thresh in enrichment_cutoffs:
            observation = query_comention_df[(query_comention_df['Count'] > min_thresh) & (query_comention_df['Count'] <= max_thresh)].shape[0]
            background = [tmp[(tmp['Count'] > min_thresh) & (tmp['Count'] <= max_thresh)].shape[0] for tmp in rand_dfs]
            rand_result_df[f"{min_thresh + 1},{max_thresh}"] = background
            # Calculate Z scores
            z_score = _calculate_z_score(observation, background)
            # Plot results
            image = _plot_results(keyword, background, observation, query, [min_thresh, max_thresh], z_score, plot_background_color, plot_query_color, plot_fontsize, plot_fontface)
            # save results in dicts for output
            enrich_results[(min_thresh, max_thresh)] = (observation, z_score)
            enrich_images[(min_thresh, max_thresh)] = image
    else:
        enrich_results, enrich_images = {}, {}

    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'PubMed_Comentions/{keyword}/')
        os.makedirs(new_savepath, exist_ok=True)
        query_comention_df.to_csv(new_savepath + f"PubMedQuery_keyword-{keyword}_field-{field}.csv")
        rand_result_df.to_csv(new_savepath + f"PubMedQueryRandomResults_keyword-{keyword}_field-{field}.csv", index = False)
        for key, value in enrich_images.items():
            value.save(new_savepath + f"PubMedQueryPlot_keyword-{keyword}_field-{field}_thresh-[>{key[0]},<={key[1]}].png")
        # Write results to file
        with open(new_savepath + f"PubMedQueryResults_keyword-{keyword}_field-{field}.txt", 'w') as f:
            for key,value in enrich_results.items():
                f.write(f">{key[0] + 1} & <={key[1]} Comentions = {value[0]} (Z = {value[1]})\n")
            f.close()
    return query_comention_df, enrich_results, enrich_images
#endregion

