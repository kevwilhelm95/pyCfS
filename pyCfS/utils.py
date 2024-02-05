"""
Utility functions for the scripts
"""

from typing import Any
from collections.abc import Iterable
from scipy.stats import hypergeom
import pkg_resources
import pandas as pd
import numpy as np

#region Statistical tests
def _hypergeo_overlap(background_size: int, query_genes:int, gs_genes:int, overlap:int) -> float:
    """
    Calculates the statistical significance (p-value) of the overlap between two gene sets
    using the hypergeometric distribution.

    This function is used to assess whether the observed overlap between a set of query genes
    and a gold standard gene set is statistically significant, given the total number of genes
    in the background set.

    Arguments:
    ---------
    background_size (int): The total number of genes in the background set.
    query_genes (int): The number of genes in the query set.
    gs_genes (int): The number of genes in the gold standard set.
    overlap (int): The number of genes that overlap between the query and gold standard sets.

    Returns:
    -------
    float: The p-value representing the statistical significance of the overlap. A lower p-value
           indicates a more significant overlap between the two gene sets.

    Note:
    -----
    The p-value is computed using the survival function ('sf') of the hypergeometric distribution
    from the SciPy stats library. The survival function provides the probability of observing
    an overlap as extreme as, or more extreme than, the observed overlap.
    """
    M = background_size
    N = query_genes
    n = gs_genes
    k = overlap
    pval = hypergeom.sf(k - 1, M, n, N)
    print(f"pval = {pval}")
    return pval
#endregion

#region Load shared background files
def _load_grch38_background(just_genes:bool = True) -> Any:
    """Return list of background genes from GRCh38

    Contains the following fields:
        chrom           object
        gene            str
        start           int64
        end             int64
    Returns
    -------
        list: List of genes annotated in GRCh38_v94
    """
    stream = pkg_resources.resource_stream(__name__, 'data/ENSEMBL-lite_GRCh38.v94.txt')
    df = pd.read_csv(stream, sep = '\t')
    if just_genes:
        return df['gene'].tolist()
    else:
        df.set_index('gene', inplace = True)
        return df

def _load_string() -> pd.DataFrame:
    """Return a dataframe of STRINGv11 protein-protein interactions

    Contains the following fields:
        node1               str
        node2               str
        neighborhood        int64
        fusion              int64
        cooccurrence        int64
        coexpression        int64
        experimental        int64
        database            int64
        textmining          int64
        combined_score      int64
    Returns
    -------
        pd.DataFrame of STRINGv11 protein interactions
    """
    stream = pkg_resources.resource_stream(__name__, 'data/STRINGv11_ProteinNames_DetailedEdges_07062022.feather')
    return pd.read_feather(stream)
#endregion

#region General cleaning and formatting
def _clean_genelists(lists: Iterable) -> list:
    """
    Normalize gene lists by ensuring all are lists and non-None.

    This function takes up to five gene sets and processes them to ensure that they are
    all list objects. `None` types are converted to empty lists, other iterables are
    converted to lists, and non-iterables are wrapped in a list.

    Parameters:
        lists (list): An iterable of lists that need to be converted to lists

    Returns:
        A list of 5 lists, corresponding to the input gene sets, sanitized toensure
        there are no None types and all elements are list objects.
    """
    clean_lists = []
    for x in lists:
        if x is None:
            # Convert None to an empty list
            clean_lists.append([])
        elif isinstance(x, Iterable) and not isinstance(x, str):
            # Convert iterables to a list, but exclude strings
            clean_lists.append(list(x))
        else:
            # Wrap non-iterables in a list
            clean_lists.append([x])
    clean_lists = [x for x in clean_lists if len(x) != 0]
    return clean_lists

def _format_scientific(value:float, threshold:float =9e-3) -> Any:
    """
    Formats a float in scientific notation if it's below a certain threshold.

    Args:
        value (float): The float value to be formatted.
        threshold (float): The threshold below which scientific notation is used. Defaults to 1e-4.

    Returns:
        str: The formatted float as a string.
    """
    if abs(value) < threshold:
        return f"{value:.1e}"
    else:
        return str(value)

def _fix_savepath(savepath:str) -> str:
    """
    Fixes the savepath by ensuring that it ends with a forward slash.

    Args:
        savepath (str): The savepath to be fixed.

    Returns:
        str: The fixed savepath.
    """
    if savepath[-1] != "/":
        savepath += "/"
    return savepath
#endregion

#region STRING network functions
def _select_evidences(evidence_lst: list, network: pd.DataFrame) -> pd.DataFrame:
    """
    Selects and returns specified columns from a network DataFrame.

    Given a list of evidence column names and a network DataFrame, this function extracts the specified columns
    along with the 'node1' and 'node2' columns from the network DataFrame and returns them as a new DataFrame.
    This is useful for filtering and focusing on specific aspects of a network represented in a DataFrame.

    Args:
        evidence_lst (list): A list of column names representing the evidence to be selected from the network DataFrame.
        network (pd.DataFrame): The network DataFrame containing at least 'node1' and 'node2' columns,
                                along with other columns that may represent different types of evidence or attributes.

    Returns:
        pd.DataFrame: A DataFrame consisting of the 'node1' and 'node2' columns from the original network DataFrame,
                      as well as the additional columns specified in evidence_lst.

    Note:
        The function assumes that the network DataFrame contains 'node1' and 'node2' columns and that all column
        names provided in evidence_lst exist in the network DataFrame. If any column in evidence_lst does not exist
        in the network DataFrame, a KeyError will be raised.
    """
    return network[['node1', 'node2'] + evidence_lst]

def _get_evidence_types(evidence_lst: list) -> list:
    """
    Processes and returns a list of evidence types for network analysis.

    This function takes a list of evidence types and, if 'all' is included in the list, replaces it with a predefined set
    of evidence types. It is primarily used to standardize and expand the evidence types used in network-based analyses.

    Args:
        evidence_lst (list): A list of strings indicating the types of evidence to be included. If the list contains 'all',
                             it is replaced with a complete set of predefined evidence types.

    Returns:
        list: A list of evidence types. If 'all' was in the original list, it is replaced by a comprehensive list of evidence types;
              otherwise, the original list is returned.

    Note:
        The predefined set of evidence types includes 'neighborhood', 'fusion', 'cooccurence', 'coexpression',
        'experimental', 'database', and 'textmining'. This function is particularly useful for initializing or
        configuring network analysis functions where a broad range of evidence types is desired.
    """
    if 'all' in evidence_lst:
        evidence_lst = ['neighborhood', 'fusion', 'cooccurence',
                        'coexpression', 'experimental', 'database', 'textmining']
    return evidence_lst

def _get_combined_score(net_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and appends a combined score for each row in a network DataFrame.

    This function takes a network DataFrame with various evidence scores and computes a combined score for each row.
    The combined score is a weighted measure based on the individual evidence scores present in the DataFrame, adjusted
    by a pre-defined probability value.

    Args:
        net_df (pd.DataFrame): A DataFrame containing network data. The first two columns are expected to be 'node1' and 'node2',
                               followed by columns representing different types of evidence scores.

    Returns:
        pd.DataFrame: The original DataFrame with an additional 'score' column appended, representing the combined score for each row.

    Note:
        The calculation of the combined score uses a fixed probability value (p = 0.041) to adjust the evidence scores.
        Scores are first normalized, then combined using a product method, and finally adjusted by the probability value.
        This function assumes that the evidence scores are represented as integers in the range 0-1000.
    """
    cols = net_df.columns.values.tolist()
    cols = cols[2:]
    p = 0.041
    for col in cols:
        net_df[col] = 1-((net_df[col]/1000) - p) / (1 -p)
        net_df[col] = np.where(net_df[col] > 1, 1, net_df[col])
    net_df['score'] = 1 - np.product([net_df[i] for i in cols], axis=0)
    net_df['score'] = net_df['score'] + p * (1 - net_df['score'])
    return net_df

def _get_edge_weight(edge_confidence:str) -> float:
    """
    Determines the weight of an edge based on its confidence level.

    This function assigns a numerical weight to an edge in a network based on a provided confidence level string.
    Different confidence levels correspond to different predefined weights.

    Args:
        edge_confidence (str): A string representing the confidence level of an edge. Accepted values are
                               'low', 'high', 'highest', and 'all'. Any other value is treated as a default case.

    Returns:
        float: The weight assigned to the edge. This is determined by the confidence level:
               - 'low' results in a weight of 0.2
               - 'high' results in a weight of 0.7
               - 'highest' results in a weight of 0.9
               - 'all' results in a weight of 0.0
               - Any other value defaults to a weight of 0.4.

    Note:
        This function is typically used in network analysis where edges have varying levels of confidence, and a numerical
        weight needs to be assigned for computational purposes.
    """
    if edge_confidence == 'low':
        weight = 0.2
    elif edge_confidence == 'high':
        weight = 0.7
    elif edge_confidence == 'highest':
        weight = 0.9
    elif edge_confidence == 'all':
        weight = 0.0
    else:
        weight = 0.4
    return weight
#endregion
