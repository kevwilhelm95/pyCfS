"""
Functions to assess variant/gene associations.

Functions:
- variants_by_sample
"""

import pandas as pd
from typing import Any
from pysam import VariantFile
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import os
from .utils import _load_grch38_background, _fix_savepath


#region VariantsBySample
def _fetch_anno(anno:Any) -> Any:
    """
    Fetches the annotation value from the given input.

    Args:
        anno (Any): The input annotation.

    Returns:
        Any: The fetched annotation value.
    """
    if isinstance(anno, tuple) and len(anno) == 1:
        return anno[0]
    else:
        return anno

def _validate_ea(ea:Any) -> float:
    """
    Checks for valid EA score
    Args:
        ea (str/float/None): EA score as string
    Returns:
        float: EA score between 0-100 if valid, otherwise returns NaN
    """
    try:
        ea = float(ea)
    except ValueError:
        if isinstance(ea, str) and (ea == 'fs-indel' or 'STOP' in ea):
            ea = 100
        else:
            ea = np.nan
    except TypeError:
        ea = np.nan
    return ea

def _fetch_ea_vep(ea:tuple, canon_ensp:str, all_ensp:tuple, csq:str, ea_parser:str) -> Any:
    """
    Fetches the EA VEP (Variant Effect Predictor) score based on the given parameters.

    Args:
        ea (list): List of EA scores.
        canon_ensp (str): Canonical Ensembl protein ID.
        all_ensp (list): List of all Ensembl protein IDs.
        csq (str): Variant consequence.
        ea_parser (str): EA parser type ('canonical', 'mean', 'max', or 'all').

    Returns:
        Any: The EA VEP score based on the given parameters.
    """
    if 'stop_gained' in csq or 'frameshift_variant' in csq or 'stop_lost' in csq or 'splice_donor_variant' in csq or 'splice_acceptor_variant' in csq:
        return 100
    if ea_parser == 'canonical':
        try:
            canon_idx = all_ensp.index(canon_ensp)
        except ValueError:
            return np.nan
        else:
            return _validate_ea(ea[canon_idx])
    else:
        new_ea = []
        for score in ea:
            new_ea.append(_validate_ea(score))
        if np.isnan(new_ea).all():
            return np.nan
        elif ea_parser == 'mean':
            return np.nanmean(new_ea)
        elif ea_parser == 'max':
            return np.nanmax(new_ea)
        else:
            return new_ea

def _convert_zygo(genotype:tuple) -> int:
    """
    Convert a genotype tuple to a zygosity integer
    Args:
        genotype (tuple): The genotype of a variant for a sample
    Returns:
        int: The zygosity of the variant (0/1/2)
    """
    if genotype in [(1, 0), (0, 1)]:
        zygo = 1
    elif genotype == (1, 1):
        zygo = 2
    else:
        zygo = 0
    return zygo

def _parse_vep(vcf_fn:str, gene:str, gene_ref:pd.DataFrame, samples:list, ea_parser:str) -> pd.DataFrame:
    """
    Parse the Variant Effect Predictor (VEP) data from a VCF file.

    Args:
        vcf (VariantFile): The VCF file object.
        gene_ref (pd.DataFrame): The gene reference data.
        contig_prefix (str): The prefix for the contig.
        samples (list): The list of samples to process.
        ea_parser (str): The EA parser.

    Returns:
        pd.DataFrame: The parsed VEP data.

    """
    # Get the vcf
    vcf = VariantFile(vcf_fn)
    # Parse for the samples only
    vcf.subset_samples(samples)
    # Get the contig type
    contig_prefix = 'chr' if 'chr' in next(vcf).chrom else ''
    contig = contig_prefix + gene_ref.chrom
    row = []
    for rec in vcf.fetch(contig=contig, start=gene_ref.start, stop=gene_ref.end):
        for sample in samples:
            zyg = _convert_zygo(rec.samples[sample]['GT'])
            rec_gene = _fetch_anno(rec.info['SYMBOL'])
            if (zyg!=0) and (rec_gene == gene):
                all_ea = rec.info.get('EA', (None,))
                all_ensp = rec.info.get('Ensembl_proteinid', (rec.info['ENSP'][0],))
                canon_ensp = _fetch_anno(rec.info['ENSP'])
                rec_hgvsp = _fetch_anno(rec.info['HGVSp'])
                csq = _fetch_anno(rec.info['Consequence'])
                ea = _fetch_ea_vep(all_ea, canon_ensp, all_ensp, csq, ea_parser=ea_parser)
                if not np.isnan(ea):
                    row.append(
                        [
                            canon_ensp,
                            rec.chrom,
                            rec.pos,
                            rec.ref,
                            rec.alts[0],
                            rec_hgvsp,
                            csq,
                            ea,
                            rec_gene,
                            sample,
                            zyg,
                            rec.info['AF'][0]
                        ]
                    )
    cols = ['ENSP', 'chr','pos','ref','alt', 'HGVSp', 'Consequence', 'EA','gene','sample','zyg','AF']
    col_type = {'chr': str, 'pos': str, 'ref': str, 'alt': str, 'sample':int, 'EA':float, 'zyg':int, 'AF':float}
    df = pd.DataFrame(row, columns = cols)
    df = df.astype(col_type)
    return df

def variants_by_sample(query:list, vcf_path:str, samples:pd.DataFrame, transcript: str = 'canonical', cores:int = 1, savepath:str = False) -> pd.DataFrame:
    """
    Retrieves variants from a VCF file for a given list of genes and sample IDs.

    Args:
        genes (list): List of genes to retrieve variants for.
        vcf_path (str): Path to the VCF file. .tbi index file must be present in the same directory.
        samples (pd.DataFrame): DataFrame containing sample IDs.
        transcript (str, optional): Transcript type to use for parsing VEP annotations. Defaults to 'canonical'.
        cores (int, optional): Number of CPU cores to use for parallel processing. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the design matrix of variants for the specified genes and samples.
    """
    # Get the sample IDs you are interested in
    sample_ids = samples.SampleID.astype(str).tolist()
    # Get the gene positions
    gene_positions = _load_grch38_background(just_genes = False)
    gene_positions = gene_positions.loc[gene_positions.index.isin(query)]

    # Parse the VCF for variants of interest
    gene_dfs = Parallel(n_jobs=cores)(delayed(_parse_vep)(
         vcf_fn=vcf_path,
         gene = gene,
         gene_ref = gene_positions.loc[gene],
         samples=sample_ids,
         ea_parser=transcript
    ) for gene in tqdm(gene_positions.index.unique()))
    design_matrix = pd.concat(gene_dfs, axis=0)

    # Map the Sample IDs to CaseControl
    design_matrix = design_matrix.merge(samples, left_on='sample', right_on='SampleID', how='left')
    design_matrix = design_matrix.drop(columns = 'SampleID')

    # Save the design matrix
    if savepath:
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'VariantsBySample/')
        os.makedirs(new_savepath, exist_ok=True)
        design_matrix.to_csv(new_savepath + 'Variants.csv', index=False)

    return design_matrix
#endregion




#region risk_prediction

#endregion


#region odds_ratios

#endregion
