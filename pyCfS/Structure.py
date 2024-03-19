"""
Structure.py
This file contains the functions for creating lollipop plots and analyzing the structures of proteins.
"""

import os
conda_env_name = os.environ.get('CONDA_DEFAULT_ENV')
r_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R')
r_libs_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R', 'library')
os.environ['R_HOME'] = r_path

import tempfile
from IPython.display import Image, display
import rpy2
from rpy2.robjects.packages import importr, isinstalled
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, globalenv
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import StrVector
import rpy2.rinterface_lib.callbacks
import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from io import BytesIO
from PIL import Image as PILImage
from .utils import _fix_savepath

warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                        message="invalid value encountered in sqrt")
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)


#region Lollipop Plots
def _filter_variants(variants: pd.DataFrame, gene: str, max_af:float, ea_lower:float, ea_upper:float) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Filters variants based on specified criteria.

    Args:
        variants (pd.DataFrame): DataFrame containing variant data.
        gene (str): Gene name to filter variants for.
        max_af (float): Maximum allele frequency threshold.
        ea_lower (float): Lower bound of effect allele frequency.
        ea_upper (float): Upper bound of effect allele frequency.

    Returns:
        tuple: A tuple containing two DataFrames - case_vars and cont_vars.
            case_vars: DataFrame containing filtered variants for case samples.
            cont_vars: DataFrame containing filtered variants for control samples.
    """
    case_vars = variants[(variants['gene'] == gene) & (variants['AF'] <= max_af) & (variants['EA'] >= ea_lower) & (variants['EA'] <= ea_upper) & (variants['CaseControl'] == 1)]
    cont_vars = variants[(variants['gene'] == gene) & (variants['AF'] <= max_af) & (variants['EA'] >= ea_lower) & (variants['EA'] <= ea_upper) & (variants['CaseControl'] == 0)]
    return case_vars, cont_vars

def _convert_amino_acids(df:pd.DataFrame, column_name:str="SUB") -> pd.DataFrame:
    """
    Convert three-letter amino acid codes to single-letter codes in a DataFrame column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the column to be converted.
        column_name (str, optional): The name of the column to be converted. Defaults to "SUB".

    Returns:
        pandas.DataFrame: The DataFrame with the converted column.
    """
    # Create a dictionary to map three-letter codes to single-letter codes
    aa_codes = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V"
    }
    
    # Replace three-letter amino acid codes with single-letter codes
    df[column_name] = df[column_name].replace(aa_codes, regex=True)
    
    return df

def _clean_variant_formats(variants: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the variant formats in the given DataFrame.

    Args:
        variants (pd.DataFrame): The DataFrame containing the variants.

    Returns:
        pd.DataFrame: The cleaned DataFrame with updated variant formats.
    """
    new_variants = variants.copy()
    # Separate HGVSp into two columns
    new_variants[['ENSP', 'SUB']] = new_variants['HGVSp'].str.split(':', expand=True)
    # Remove leading 'p.' from SUB
    new_variants['SUB'] = new_variants['SUB'].str.replace('p.', '')
    # Convert three-letter amino acid codes to single-letter codes
    new_variants = _convert_amino_acids(new_variants)
    # Remove NA rows
    new_variants = new_variants.dropna()
    # Aggregate Zyg
    new_variants = new_variants.groupby(['SUB', 'EA']).agg(
        ENSP = ('ENSP', 'first'),
        SUB = ('SUB', 'first'),
        EA = ('EA', 'first'),
        AC = ('zyg', 'sum')
    )
    new_variants = new_variants.reset_index(drop=True)
    return new_variants

def _fishers_exact_test(case_vars: pd.DataFrame, cont_vars: pd.DataFrame, case_pop:int, cont_pop:int) -> (float, float, float, float): # type: ignore
    """
    Perform Fisher's exact test to calculate odds ratio, confidence interval, and p-value.

    Args:
        case_vars (pd.DataFrame): DataFrame containing case variables.
        cont_vars (pd.DataFrame): DataFrame containing control variables.
        case_pop (int): Total population of cases.
        cont_pop (int): Total population of controls.

    Returns:
        tuple: A tuple containing the odds ratio, lower confidence interval, upper confidence interval, and p-value.
    """
    # Create contingency table
    obs_case_alleles = case_vars['zyg'].sum()
    obs_cont_alleles = cont_vars['zyg'].sum()
    exp_case_alleles = case_pop - obs_case_alleles
    exp_cont_alleles = cont_pop - obs_cont_alleles
    contingency_table = [[obs_case_alleles, obs_cont_alleles], [exp_case_alleles, exp_cont_alleles]]
    
    # Perform Fisher's exact test
    oddsratio_data = sm.stats.Table2x2(contingency_table)
    odds_ratio = oddsratio_data.oddsratio
    lower_ci, upper_ci = oddsratio_data.oddsratio_confint()
    pval = oddsratio_data.oddsratio_pvalue()
    
    return odds_ratio, lower_ci, upper_ci, pval

def _r_install_package(package_name:str):
    """
    Attempts to install an R package and its dependencies.
    
    Parameters:
    - package_name: The name of the package to install.
    """
    # Define the CRAN repository URL
    cran_mirror = 'https://cran.rstudio.com/'
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1)  # Select the first mirror in the list
    os.environ['R_REMOTES_NO_ERRORS_FROM_WARNINGS'] = 'true'
    os.environ['R_REMOTES_UPGRADE'] = "never"

    # Install the package if it is not already installed
    if not isinstalled(package_name):
        try:
            if package_name == 'EvoTrace':
                remote = importr('remotes')
                remote.install_github('LichtargeLab/EvoTrace', build_vignettes=False)
            else:
                utils.install_packages(StrVector([package_name]))
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            print(f"An error occurred while installing {package_name}: {e}")
            # Attempt to install system dependencies if the package is known to require compilation
            if package_name in ['interp']:
                print(f"Attempting to install system dependencies for {package_name}")
                try:
                    utils.install_packages(StrVector([package_name]))
                except Exception as e:
                    print(f"Failed to install {package_name} after attempting to resolve dependencies: {e}")

def _r_lollipop_plot(case_vars: pd.DataFrame, cont_vars: pd.DataFrame, plot_domain:bool, ac_scale:str, ea_color:str, domain_min_dist:int) -> PILImage.Image:
    """
    Generate a lollipop plot using R's EvoTrace package.

    Args:
        case_vars (pd.DataFrame): DataFrame containing case variants.
        cont_vars (pd.DataFrame): DataFrame containing control variants.
        plot_domain (bool): Flag indicating whether to plot the protein domain.
        ac_scale (str): Scale for the allele count axis. Must be one of 'linear' or 'log'.
        ea_color (str): Color scheme for the effect allele. Must be one of 'prismatic', 'gray_scale', 'EA_bin', or 'black'.
        domain_min_dist (int): Minimum distance between two domains.

    Returns:
        PIL.Image.Image: The generated lollipop plot as a PIL Image object.
    """
    # Set CRAN mirror
    cran_mirror = "https://cran.rstudio.com"  # Set your desired CRAN mirror URL here
    robjects.r.options(repos=cran_mirror)
    # Install R packages
    _r_install_package('remotes')
    _r_install_package('EvoTrace')
    evotrace = importr('EvoTrace')
    # Convert local DataFrames to R DataFrames
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_case_vars = robjects.conversion.py2rpy(case_vars)
        r_cont_vars = robjects.conversion.py2rpy(cont_vars)
    # Check options
    prot_id = case_vars.loc[0, 'ENSP']
    if ea_color not in ['prismatic', 'gray_scale', 'EA_bin', 'black']:
        raise ValueError("ea_color must be one of 'prismatic', 'gray_scale', 'EA_bin', or 'black'")
    if ac_scale not in ['linear', 'log']:
        raise ValueError("ac_scale must be one of 'linear' or 'log'")
    # Create temporary directory
    temp_file = tempfile.NamedTemporaryFile(delete = False, suffix = ".png")
    plot_path = temp_file.name
    # Create the lollipop plot
    globalenv['plot'] = evotrace.LollipopPlot2(
        variants_case = r_case_vars,
        variants_ctrl = r_cont_vars,
        prot_id = prot_id,
        plot_domain = plot_domain,
        AC_scale = ac_scale,
        show_EA_bin = True,
        fix_scale = True,
        EA_color = ea_color, 
        domain_min_dist = domain_min_dist
    )
    r(f"""ggsave("{plot_path}", plot, device = "png", width = 10, height = 5, dpi = 300)""")
    # Display the plot
    lollipop_plot_plot = Image(filename=plot_path)
    os.unlink(plot_path)
    temp_file.close()
    image_buffer = BytesIO(lollipop_plot_plot.data)
    lollipop_plot_plot = PILImage.open(image_buffer)
    return lollipop_plot_plot

def lollipop_plot(variants: pd.DataFrame, gene: str, case_pop:int=0, cont_pop:int=0, max_af:float = 1.0, ea_lower:float = 0.0, ea_upper:float = 100.0, show_domains:bool = True, ac_scale:str = 'linear', ea_color:str = 'prismatic', domain_min_dist:int = 20, savepath:str = False) -> (Image, float, float, float, float): # type: ignore
    """
    Generate a lollipop plot for a given gene based on variant data.

    Parameters:
    - variants (pd.DataFrame): DataFrame containing variant data.
    - gene (str): Gene of interest.
    - case_pop (int, optional): Number of cases in the variant file. If not provided, it will be calculated based on the data.
    - cont_pop (int, optional): Number of controls in the variant file. If not provided, it will be calculated based on the data.
    - max_af (float, optional): Maximum allele frequency threshold for filtering variants. Default is 1.0.
    - ea_lower (float, optional): Lower bound for effect size threshold. Default is 0.0.
    - ea_upper (float, optional): Upper bound for effect size threshold. Default is 100.0.
    - show_domains (bool, optional): Whether to show domain annotations on the lollipop plot. Default is True.
    - ac_scale (str, optional): Scale for the allele count axis. Default is 'linear'.
    - ea_color (str, optional): Color scheme for effect size. Default is 'prismatic'.
    - domain_min_dist (int, optional): Minimum distance between domain annotations. Default is 20.
    - savepath (str, optional): Path to save the lollipop plot image. If not provided, the plot will not be saved.

    Returns:
    - plot (Image): Lollipop plot image.
    - pval (float): p-value from Fisher's exact test.
    - odds_ratio (float): odds ratio from Fisher's exact test.
    - lower_ci (float): lower confidence interval from Fisher's exact test.
    - upper_ci (float): upper confidence interval from Fisher's exact test.
    """
    # Check the number of samples
    if (case_pop == 0) & (cont_pop == 0):
        case_pop = variants['sample'][variants['CaseControl'] == 1].nunique()
        cont_pop = variants['sample'][variants['CaseControl'] == 0].nunique()
    print(f"Number of cases in variant file: {case_pop}")
    print(f"Number of controls in variant file: {cont_pop}")
    # Filter dataframe for gene and split by case and control
    case_vars, cont_vars = _filter_variants(variants, gene, max_af, ea_lower, ea_upper)
    # Clean variant annotations
    case_vars_collapsed, cont_vars_collapsed = _clean_variant_formats(case_vars), _clean_variant_formats(cont_vars)
    # Calculate fisher's exact test
    odds_ratio, lower_ci, upper_ci, pval = _fishers_exact_test(case_vars, cont_vars, case_pop, cont_pop)
    # Create the lollipop plot
    plot = _r_lollipop_plot(case_vars_collapsed, cont_vars_collapsed, plot_domain = show_domains, ac_scale = ac_scale, ea_color = ea_color, domain_min_dist = domain_min_dist)
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, f'LollipopPlots/{gene}/')
        os.makedirs(new_savepath, exist_ok=True)
        plot.save(new_savepath + f'{gene}_lollipop_plot.png')
    # Need to group by zyg for fishers exact test
    return plot, pval, odds_ratio, lower_ci, upper_ci
#endregion