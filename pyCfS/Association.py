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
# Risk Prediction
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import DeltaYStopper
from sklearn.metrics import balanced_accuracy_score, roc_curve, RocCurveDisplay, auc
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFE, RFECV

import numpy as np
import statsmodels.api as sm
from scipy.stats import ks_2samp
import math
import matplotlib.pyplot as plt
import io
import joblib
from PIL import Image
import os
from .utils import _load_grch38_background, _fix_savepath

# Ignore machine learning warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"


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
def _validate_sample_dfs(sample_df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
    """
    Validate the sample DataFrames for the risk prediction function. Sample DataFrames must have sample_ids as the index and column names as 'SampleID' and 'CaseControl'.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame.

    Returns:
        pd.DataFrame: The validated sample DataFrame.
    """
    # Check number of columns
    if len(sample_df.columns) != 2:
        raise ValueError(f"{sample_type} DataFrame must have 2 columns. First column is sample IDs and second column is CaseControln (1 = Case, 0 = Control).")
    # Check for valid 'CaseControl' values
    if not sample_df.iloc[:, 1].isin([0, 1]).all():
        raise ValueError(f"{sample_type} - CaseControl values must be 0 or 1.")
    # Change the column values for later accession
    sample_df.columns = ['SampleID', 'CaseControl']
    return sample_df

def _validate_feature_matrix(feature_matrix: pd.DataFrame, train_samples: pd.DataFrame, test_samples: pd.DataFrame) -> None:
    """
    Validate the feature matrix for the risk prediction function. Feature matrix must have sample_ids as the index and column names as gene names or feature names.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix.

    Returns:
        pd.DataFrame: The validated feature matrix.
    """
    # Check train samples are in feature matrix index
    if not train_samples.iloc[:, 0].isin(feature_matrix.index).all():
        missing_samples = train_samples[~train_samples.iloc[:, 0].isin(feature_matrix.index)].iloc[:, 0].tolist()
        raise ValueError(f"Train samples are not in feature matrix index. {len(missing_samples)} samples in test_samples not found in feature_matrix: {missing_samples}")
    # Check test samples are in feature matrix index
    if not test_samples.iloc[:, 0].isin(feature_matrix.index).all():
        # Which sample IDs in test_samples are not in feature matrix
        missing_samples = test_samples[~test_samples.iloc[:, 0].isin(feature_matrix.index)].iloc[:, 0].tolist()
        raise ValueError(f"Test samples are not in feature matrix index. {len(missing_samples)} samples in test_samples not found in feature_matrix: {missing_samples}")

def _generate_x_y_matrices(feature_matrix: pd.DataFrame, sample_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame): # type: ignore
    """
    Generate the X and y matrices for the risk prediction function.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix.
        sample_df (pd.DataFrame): The sample DataFrame.

    Returns:
        tuple: A tuple containing the X and y matrices.
    """
    # Merge the dataframes
    sub_df = sample_df.merge(feature_matrix, left_on='SampleID', right_index=True, how = 'inner')
    sub_df.index = sub_df.SampleID
    # Get the X matrix
    x = sub_df.drop(columns = ['SampleID', 'CaseControl'])
    # Get the y matrix
    y = sub_df.CaseControl
    return x, y

def _validate_separate_input_models(input_models: Any) -> list:
    """
    Validates the input models and returns a list of valid models.

    Args:
        input_models (Union[str, list]): The input models to be validated. It can be a string or a list of strings.

    Returns:
        list: A list of valid models.

    Raises:
        ValueError: If the input models are neither a string nor a list of strings.

    """
    if isinstance(input_models, str):
        # Check if the input has comma separated values
        if ', ' in input_models:
            input_models = input_models.split(', ')
        elif "," in input_models:
            input_models = input_models.split(',')
        else:
            input_models = [input_models]
    elif isinstance(input_models, list):
        pass
    else:
        raise ValueError("Input models must be a string or list of strings")

    # Check against valid models
    valid_models = ['LR', 'SVC', 'GB', 'RF', 'XGB']
    for model in input_models:
        if model not in valid_models:
            print(f"Invalid model: {model}. Skipping this model. Valid models are: {valid_models}.")
            input_models.remove(model)
    return input_models

def _get_hyperparameter_search_space(model: str, n_features:int) -> (dict, Any): # type: ignore
    """
    Get the hyperparameter search space for the given model and number of features.

    Args:
        model (str): The model name. Possible values: 'LR', 'SVC', 'GB', 'RF', 'XGB'.
        n_features (int): The number of features.

    Returns:
        tuple: A tuple containing the hyperparameter search space dictionary and the corresponding estimator object.
    """

    params = dict()
    # Logistic Regression
    if model == 'LR':
        params['penalty'] = Categorical(['l1', 'l2', 'elasticnet'])
        # 'newton-cg', 'lbfgs', 'liblinear', 'sag'
        params['solver'] = Categorical(['saga'])
        params['max_iter'] = Categorical([5000, 10000])
        params['warm_start'] = Categorical([True, False])
        params['l1_ratio'] = Real(0.0001, 0.5, prior='log-uniform')
        params['tol'] = Real(0.0001, 100.0, prior='log-uniform')
        params['C'] = Real(0.0001, 100.0, prior='log-uniform')
        params['random_state'] = [n_features]
        estimator = LogisticRegression()
    # Support Vector Machine
    if model == 'SVC':
        # Lower bound - 1e-6
        params['C'] = Real(0.0001, 10.0, prior='log-uniform')
        # Lower bound - 1e-6
        params['gamma'] = Real(0.0001, 1.0, prior='log-uniform')
        params['degree'] = Integer(1.0, 5.0)
        params['kernel'] = Categorical(['linear', 'poly', 'rbf'])  # 'sigmoid'
        params['probability'] = Categorical([True])
        params['random_state'] = [n_features]
        estimator = SVC()
    # Gradient Boosting
    if model == 'GB':
        params['n_estimators'] = Integer(100, 10000, prior='log-uniform')
        params['loss'] = Categorical(['log_loss','exponential'])
        params['learning_rate'] = Real(0.0001, 1.0, prior='log-uniform')
        params['subsample'] = Real(0.0001, 0.5, prior='log-uniform')
        params['criterion'] = Categorical(['friedman_mse', 'squared_error'])
        params['max_depth'] = Integer(1, round(np.sqrt(n_features), 0), prior = 'log-uniform')
        params['warm_start'] = Categorical([True, False])
        params['tol'] = Real(0.0001, 100.0, prior='log-uniform')
        params['ccp_alpha'] = Real(0.0001, 100.0, prior='log-uniform')
        params['random_state'] = [n_features]
        estimator = GradientBoostingClassifier()
    if model == 'RF':
        params['n_estimators'] = Integer(100, 10000, prior='log-uniform')
        params['criterion'] = Categorical(['gini', 'entropy'])
        params['bootstrap'] = Categorical([True])
        params['warm_start'] = Categorical([True, False])
        params['ccp_alpha'] = Real(0.0001, 100.0, prior='log-uniform')
        params['oob_score'] = Categorical([True, False])
        params['min_samples_split'] = Real(0.001, 0.5, prior='log-uniform')
        params['min_samples_leaf'] = Integer(1, 10)
        params['max_depth'] = Integer(1.0, round(np.sqrt(n_features), 0), prior = 'log-uniform')
        params['max_features'] = Categorical(['sqrt', 'log2'])
        params['random_state'] = [n_features]
        estimator = RandomForestClassifier()
    if model == 'XGB':
        params['n_estimators'] = Integer(10, 5000, prior='log-uniform')
        params['subsample'] = Real(0.1, 0.5, prior = 'log-uniform')
        params['max_depth'] = Integer(2, round(np.sqrt(n_features),0))
        params['eta'] = Real(0.01, 0.5, prior='log-uniform')
        #params['gamma'] = Real(1e-4, 0.5, prior = 'log-uniform') -- Warning
        params['reg_alpha'] = Real(1e-4, 0.5, prior='log-uniform')
        params['reg_lambda'] = Real(1e-4, 0.9, prior='log-uniform')
        params['booster'] = Categorical(['dart', 'gbtree'])
        params['tree_method'] = Categorical(['hist'])
        params['random_state'] = [n_features]
        estimator = xgb.XGBClassifier()
    return params, estimator

def _aggregate_feature_selection(cv_obj: Any, estimator: Any, x_train:pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate feature selection results from cross-validation and the final estimator.

    Args:
        cv_obj (Any): The cross-validation object used for feature selection.
        estimator (Any): The final estimator used for feature selection.
        x_train (pd.DataFrame): The training data.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated feature selection results.

    """
    # Get feature importances from the final estimator
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Feature_Importance': importances})
    elif hasattr(estimator, 'coef_'):
        importances = estimator.coef_[0]
        feature_importances = pd.DataFrame({'Feature': x_train.columns, 'Feature_Importance': importances})
    # Trying out creating a saving object
    rfecv_results = pd.DataFrame({
        'Feature': cv_obj.feature_names_in_,
        'Support': cv_obj.support_,
        'Ranking': cv_obj.ranking_
    })
    # Merge the results
    feature_importances = feature_importances.merge(rfecv_results, left_on='Feature', right_on='Feature', how='left')
    feature_importances = feature_importances.sort_values(by=['Feature_Importance', 'Ranking'], ascending=False).reset_index(drop=True)
    return feature_importances

def _plot_rfe_cv(rfe_cv_results: pd.DataFrame) -> Image:
    """
    Plot the results of the Recursive Feature Elimination Cross-Validation.

    Args:
        rfe_cv_results (pd.DataFrame): The results of the RFE-CV.

    Returns:
        PIL.Image.Image: The image of the plot.

    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        rfe_cv_results.n_features,
        rfe_cv_results.mean_roc_auc,
        yerr=rfe_cv_results.std_roc_auc,
        label='ROC-AUC',
        color='r',
        linewidth=2,
        ecolor = 'gray',
        elinewidth = 1,
        capsize = 3)
    plt.xlabel('Number of Features')
    plt.ylabel('ROC-AUC')
    plt.title('Recursive Feature Elimination Cross-Validation')
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _rfe_cv(estimator: Any, x_train: pd.DataFrame, y_train: pd.DataFrame, min_feature_ratio:float, cores: int) -> (list, pd.DataFrame, pd.DataFrame, Image): # type: ignore
    """
    Perform recursive feature elimination with cross-validation (RFECV) to select the optimal features for a given estimator.

    Parameters:
        - estimator (Any): The estimator object implementing the 'fit' method.
        - x_train (pd.DataFrame): The input features for training.
        - y_train (pd.DataFrame): The target variable for training.
        - min_feature_ratio (float): The minimum ratio of features to select.
        - cores (int): The number of CPU cores to use for parallel processing.

    Returns:
        - selected_features (list): The list of selected features.
        - feature_importances (pd.DataFrame): The feature importances.
        - rfe_cv_results (pd.DataFrame): The results of RFECV.
        - rfe_cv_plot (matplotlib.pyplot): The plot of RFECV results.
    """
    # Initialize the RFECV object
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=len(x_train.columns))
    min_features = math.floor(min_feature_ratio * x_train.shape[1])
    rfecv = RFECV(
        estimator,
        step = 1,
        cv = cv,
        scoring = 'roc_auc',
        min_features_to_select = min_features,
        verbose = 0,
        n_jobs = cores,
        importance_getter = 'auto'
    )
    # Fit the RFECV object
    rfecv = rfecv.fit(x_train, y_train)
    estimator.fit(x_train, y_train)
    # Get the feature importances
    feature_importances = _aggregate_feature_selection(rfecv, estimator, x_train)
    selected_features = feature_importances.loc[feature_importances.Support == True, 'Feature'].tolist()

    # Collect RFE results
    rfe_cv_results = pd.DataFrame({
        'n_features': range(min_features, x_train.shape[1] + 1),
        'mean_roc_auc': rfecv.cv_results_['mean_test_score'],
        'std_roc_auc': rfecv.cv_results_['std_test_score']
    })
    rfe_cv_plot = _plot_rfe_cv(rfe_cv_results)

    return selected_features, feature_importances, rfe_cv_results, rfe_cv_plot

def _old_run_recursive_feature_elimination(estimator: Any, x_train: pd.DataFrame, y_train: pd.DataFrame, n_features:int, max_feature_ratio:float) -> (Any, pd.DataFrame): #type: ignore
    """
    Run Recursive Feature Elimination (RFE) on the given model.

    Args:
        estimator (Any): The model estimator.
        x_train (pd.DataFrame): The training feature matrix.
        y_train (pd.DataFrame): The training target vector.
        n_features (int): The number of features to select.

    Returns:
        tuple: A tuple containing the RFE object and the feature matrix with the selected features.
    """
    # Initialize holding objects and settings
    meta_df = pd.DataFrame(index = x_train.columns)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=len(x_train.columns))
    # Loop through cross-validation
    for i, (train_index, test_index) in enumerate(cv.split(x_train, y_train)):
        print("Fold: ", i)
        # Get cross-validation groups
        x_train_cv, x_test_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
        # Initialize RFE
        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(x_train_cv, y_train_cv)
        # Get the feature ranking
        rfe_rank = pd.DataFrame(
            {f"Ranking-Fold-{i}": rfe.ranking_,},
            index = rfe.feature_names_in_
        ).sort_values(by=f"Ranking-Fold-{i}", ascending=True)
        # Merge with meta_df
        meta_df = meta_df.merge(rfe_rank, left_index=True, right_index=True, how='outer')
        print(meta_df.head(3))
        meta_df = meta_df.sort_values(by = f"Ranking-Fold-{i}", ascending = False)
        print(meta_df.head(3))

        # Get the accuracy of genes for prediction against cross-validation
        top_features = []
        for gene, row in rfe_rank.iterrows():
            top_features.append(gene)
            x_train_cv_sub = x_train_cv[top_features]
            x_test_cv_sub = x_test_cv[top_features]
            estimator.fit(x_train_cv_sub, y_train_cv)
            y_pred = estimator.predict(x_test_cv_sub)
            meta_df.loc[gene, f"Accuracy-Fold-{i}"] = balanced_accuracy_score(y_test_cv, y_pred)

    # Calculate averages across meta-df
    meta_df['Mean_Rank'] = meta_df[[x for x in meta_df.columns if "Ranking" in x]].mean(axis=1)
    meta_df['Mean_Accuracy'] = meta_df[[x for x in meta_df.columns if "Accuracy" in x]].mean(axis=1)
    meta_df['Accuracy_Std'] = meta_df[[x for x in meta_df.columns if "Accuracy" in x]].std(axis=1)
    meta_df = meta_df.sort_values(by='Mean_Rank', ascending=True)
    print(meta_df.head(3))
    #### Need to output and save meta-df

    # Calculate the cutoff point for features
    max_n_features = math.ceil(meta_df.shape[0] * max_feature_ratio)
    cutoff = meta_df.loc[0:int(max_n_features), 'Mean_Accuracy'].idxmax()
    cutoff = meta_df.index.get_loc(cutoff)

def _bayesian_optimization_auroc_progression(df: pd.DataFrame) -> Image:
    """
    Plot the progression of AUROC through the hyperparameter search space.

    Args:
        df (pd.DataFrame): The DataFrame containing the AUROC scores.

    Returns:
        PIL.Image.Image: The image of the plot.

    """
    plt.plot(df.index, df.mean_test_score)
    plt.xlabel("HyperParameter Search Index", size=12)
    plt.ylabel("AUROC", size=12)
    plt.title("Progression of AUROC through hyperparameter search space", size=14)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _optimize_hyperparameters(estimator: Any, params: dict, x_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int, n_splits: int, n_repeats: int, cores: int) -> (Any, pd.DataFrame, Image): #type: ignore
    """
    Optimize hyperparameters using Bayesian optimization.

    This function performs hyperparameter optimization using Bayesian optimization
    with a custom stopper callback. It uses the BayesSearchCV class from the scikit-optimize
    library to perform the optimization.

    Parameters
    ----------
    estimator : Any
        The estimator object to be optimized.

    params : dict
        The search space for the hyperparameters.

    x_train : pd.DataFrame
        The input training data.

    y_train : pd.DataFrame
        The target training data.

    n_iter : int
        The number of iterations for the optimization.

    n_splits : int
        The number of splits for cross-validation.

    n_repeats : int
        The number of repeats for cross-validation.

    cores : int
        The number of CPU cores to use for parallel processing.

    Returns
    -------
    tuple
        A tuple containing the best score, best parameters, cross-validation results,
        and a plot of the optimization progression.

    """
    # Define our custom stopper for the BayesSearchCV
    class CustomDeltaYStopper(DeltaYStopper):
        """
        Custom callback for the BayesSearchCV function to stop the search early when
        the best scores have converged below a specified threshold.

        Parameters
        ----------
        delta : float, required
            The threshold value for the difference between the best scores. If the
            difference between the best n_best scores is less than delta for
            patience consecutive iterations, the search will stop early.

        n_best : int, optional, default: 5
            The number of best scores to consider when checking for convergence.
            The difference between the best n_best scores will be compared to the
            delta value.

        patience : int, optional, default: 5
            The number of consecutive iterations for which the difference between
            the best n_best scores must be less than delta to stop the search early.

        Attributes
        ----------
        _count : int
            A counter to track the number of consecutive iterations for which the
            difference between the best n_best scores is less than delta.

        Methods
        -------
        _criterion(result) -> bool
            A helper function that checks if the search should be stopped early
            based on the provided convergence criteria.

        """
        def __init__(self, delta:float, n_best:int=5, patience:int=5, min_iters:int = 20) -> None:
            super().__init__(delta)
            self.n_best = n_best
            self.patience = patience
            self._count = 0
            self.min_iters = min_iters

        def _criterion(self, result: Any) -> bool:
            """
            Check if the criterion for early stopping is met.

            Args:
                result (object): The result object containing function values.

            Returns:
                bool: True if the criterion is met and early stopping should occur, False otherwise.
            """
            if len(result.func_vals) < self.min_iters:
                return False
            if len(result.func_vals) < self.n_best:
                return False
            best_scores = sorted(result.func_vals)[-self.n_best:]
            delta = best_scores[-1] - best_scores[0]
            if delta < self.delta:
                self._count += 1
                if self._count >= self.patience:
                    print(f"Early stopping occurred at iteration: {len(result.func_vals)}")
                    return True
            else:
                self._count = 0

            return False

    # Initialize search objects
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=len(x_train.columns))
    delta_y_callback = CustomDeltaYStopper(delta=0.0001, n_best = 5, patience = 5, min_iters = 20)
    search = BayesSearchCV(
        estimator = estimator,
        search_spaces = params,
        n_jobs = cores,
        n_iter = n_iter,
        cv = cv,
        scoring = 'roc_auc',
        verbose = 0,
        random_state = len(x_train.columns)
    )
    search.fit(x_train, y_train, callback = delta_y_callback)
    cv_result = pd.DataFrame(search.cv_results_)
    cv_result_plot = _bayesian_optimization_auroc_progression(cv_result)

    return search.best_score_, search.best_params_, cv_result, cv_result_plot

def _set_hyperparameters(estimator: Any, params: dict) -> Any:
    """
    Set the hyperparameters for the given model.

    Args:
        model (Any): The model object.
        params (dict): The hyperparameters.

    Returns:
        Any: The model object with the set hyperparameters.
    """
    estimator.set_params(**params)
    return estimator

def _optimize_features_hyperparameters(input_models: Any, rfe:bool, x_train:pd.DataFrame, y_train:pd.DataFrame, max_feature_ratio:float, cores:int) -> (Any, pd.DataFrame, dict, dict, dict, dict, dict): # type: ignore
    """
    Optimize features and hyperparameters for each model.

    Args:
        input_models (Any): The input models to test.
        rfe (bool): Flag indicating whether to perform Recursive Feature Elimination (RFE).
        x_train (pd.DataFrame): The training data features.
        y_train (pd.DataFrame): The training data labels.
        max_feature_ratio (float): The maximum feature ratio for RFE.
        cores (int): The number of cores to use for parallel processing.

    Returns:
        tuple: A tuple containing the final estimator, best results dataframe, RFE importances, RFE results, RFE results plot, Bayesian results, and Bayesian results plot.
    """
    # Get the estimators to test
    models_to_test = _validate_separate_input_models(input_models)
    # Set holder values
    best_results = dict()
    best_params_dict = dict()
    rfe_importances, rfe_results, rfe_results_plot = dict(), dict(), dict()
    bayes_results, bayes_results_plot = dict(), dict()
    # Optimize Hyperparameters for each model
    for model in models_to_test:
        ### Need to understand how to save everything for each model
        params, estimator = _get_hyperparameter_search_space(model, x_train.shape[1])
        # Perform rfe if needed
        if rfe:
            selected_features, feature_importances, rfe_cv_results, rfe_cv_plot = _rfe_cv(estimator, x_train, y_train, max_feature_ratio, cores)
            rfe_importances[model] = feature_importances
            rfe_results[model] = rfe_cv_results
            rfe_results_plot[model] = rfe_cv_plot
            x_train = x_train[selected_features]
        else:
            rfe_importances[model] = None
            rfe_results[model] = None
            rfe_results_plot[model] = None
            selected_features = x_train.columns.to_list()

        # Perform Bayesian hyperparameter optimization
        params, estimator = _get_hyperparameter_search_space(model, len(selected_features))
        best_score, best_params, cv_result, cv_plot = _optimize_hyperparameters(estimator, params, x_train, y_train, n_iter=50, n_splits=5, n_repeats=5, cores=cores)
        best_results[model] = best_score
        best_params_dict[model] = best_params
        bayes_results[model] = cv_result
        bayes_results_plot[model] = cv_plot
    # Merge best results and params for saving
    best_result_df = pd.DataFrame({'Model': list(best_results.keys()), 'Best_AUROC': list(best_results.values())}).sort_values(by='Best_AUROC', ascending=False).reset_index(drop=True)
    best_params_df = pd.DataFrame.from_dict(best_params_dict, orient='index').reset_index().rename(columns={'index': 'Model'})
    best_df = best_result_df.merge(best_params_df, left_on='Model', right_on='Model', how='left')

    # Set the best model
    best_model = best_result_df.loc[0, 'Model']
    model_params = best_params_dict[best_model]
    final_estimator = _set_hyperparameters(estimator, model_params)
    final_estimator.fit(x_train, y_train)

    return final_estimator, best_model, selected_features, best_df, rfe_importances, rfe_results, rfe_results_plot, bayes_results, bayes_results_plot, models_to_test

def _save_feature_weights(estimator: Any, model: str, feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Save the feature weights of a given model.

    Parameters:
    - estimator (Any): The trained model estimator.
    - model (str): The type of model used. Possible values are 'LR', 'SVC', 'GB', 'RF', 'XGB'.
    - feature_matrix (pd.DataFrame): The feature matrix used for training the model.

    Returns:
    - pd.DataFrame: A DataFrame containing the feature weights sorted in descending order.

    Note:
    - For models 'LR' and 'SVC', the function retrieves the coefficients of the model and maps them to the feature names.
    - For models 'GB', 'RF', and 'XGB', the function retrieves the feature importances directly from the model.
    - If the model is not one of the specified types, the function returns None.
    """
    if model in ['LR', 'SVC']:
        try:
            coefficients = estimator.coef_[0]
            feature_weights = dict(zip(feature_matrix.columns, coefficients))
        except:
            print("Model does not have coefficients")
    elif model in ['GB', 'RF', 'XGB']:
        importances = estimator.feature_importances_
        feature_weights = dict(zip(feature_matrix.columns, importances))
    else:
        feature_weights = None
        return feature_weights

    if feature_weights:
        feature_weights_df = pd.DataFrame.from_dict(feature_weights, orient='index', columns=['Feature Weight'])
        feature_weights_df = feature_weights_df.sort_values(by='Feature Weight', ascending=False)
        return feature_weights_df

def _plot_auroc(fpr: pd.Series, tpr: pd.Series, roc_auc:float) -> Image:
    """
    Plots the Area Under the Receiver Operating Characteristic (AUROC) curve for a given estimator.

    Parameters:
        estimator (Any): The trained estimator object.
        x_test (pd.DataFrame): The input features for testing.
        y_test (pd.DataFrame): The target labels for testing.

    Returns:
        Image: The AUROC curve plot as an Image object.
    """
    plt.clf()
    plt.close('all')
    _ = plt.figure(figsize=(5, 5))
    display = RocCurveDisplay(fpr = fpr, tpr=tpr, roc_auc = roc_auc, estimator_name=' ')
    display.plot()
    plt.legend(loc='upper left')
    plt.xlabel('')
    plt.ylabel('')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)
    plt.close()
    return image

def _plot_histo_w_or(predictions_gene: pd.DataFrame, bins:int = 50, xlim_lower:float = None, xlim_upper:float = None, ylim:tuple = (None, None), ylim_or: tuple = (None, None)) -> (Image, pd.DataFrame): # type: ignore
    """
    Plots a histogram of the probability of being a case or control, along with the odds ratio (OR) values.

    Args:
        predictions_gene (pd.DataFrame): A DataFrame containing the predictions for each sample, including the 'case' column indicating whether the sample is a case or control.
        bins (int, optional): The number of bins for the histogram. Defaults to 50.
        xlim_lower (float, optional): The lower limit of the x-axis range for the histogram. Defaults to None.
        xlim_upper (float, optional): The upper limit of the x-axis range for the histogram. Defaults to None.
        ylim (tuple, optional): The y-axis limits for the histogram. Defaults to (None, None).
        ylim_or (tuple, optional): The y-axis limits for the odds ratio (OR) plot. Defaults to (None, None).

    Returns:
        tuple: A tuple containing the histogram image (Image) and a DataFrame (pd.DataFrame) with the odds ratio (OR) values for different thresholds.

    Raises:
        None

    """
    def calculate_or_with_ci(tp:int, fp:int, fn:int, tn:int) -> tuple:
        """
        :param a: affected with mutation
        :param b: healthy with mutation
        :param c: affected w/o mutation
        :param d: healthy w/o mutation
        :return: odds ratio, p-value, and confidence interval
        """
        table = np.zeros((2, 2))
        table[0][0] = tp
        table[0][1] = fp
        table[1][0] = fn
        table[1][1] = tn
        or_table = sm.stats.Table2x2(table)
        odds_ratio = or_table.oddsratio
        p_value = or_table.oddsratio_pvalue()
        confidence_interval = list(or_table.oddsratio_confint())
        '''used for cleaned-up version'''
        confidence_interval = [float(f'{x:.2f}') for x in confidence_interval]
        return odds_ratio, p_value, confidence_interval

    thrlist = np.arange(0.5, 1.0, 0.05).tolist()
    thrlist = np.arange(0.5, 0.6, 0.01).tolist()
    or_df = pd.DataFrame(columns=['Thr', 'OR', 'CI', 'pval',
                                  'Dis w exp', 'HC w exp', 'Dis w/o exp', 'HC w/o exp'])
    for i, thr in enumerate(thrlist):
        high_thr = thr
        low_thr = 1-thr
        # high = gene_probs[gene_probs['Prob']>high_thr].copy()
        # low = gene_probs[gene_probs['Prob']<low_thr].copy()
        if thr >=0.5:
            high = predictions_gene[predictions_gene['Case'] > high_thr].copy()
            low = predictions_gene[predictions_gene['Case'] < low_thr].copy()
        elif thr < 0.5:
            high = predictions_gene[predictions_gene['Control'] > high_thr].copy()
            low = predictions_gene[predictions_gene['Control'] < low_thr].copy()
        TP = np.sum(high['True_Class'])
        FP = high.shape[0]-TP
        FN = np.sum(low['True_Class'])
        TN = low.shape[0]-FN

        odr, p, ci = calculate_or_with_ci(TP, FP, FN, TN)
        or_df.loc[i, 'Thr'] = thr
        or_df.loc[i, 'OR'] = odr
        or_df.loc[i, 'CI'] = ci
        or_df.loc[i, 'pval'] = p
        or_df.loc[i, 'Dis w exp'] = TP
        or_df.loc[i, 'HC w exp'] = FP
        or_df.loc[i, 'Dis w/o exp'] = FN
        or_df.loc[i, 'HC w/o exp'] = TN

    no_gene = []
    for th in thrlist:
        try:
            TP = np.where(
                (predictions_gene['True_Class'] == 1) &
                (predictions_gene['Case'] > th)
            )
            TN = np.where(
                (predictions_gene['True_Class'] == 0) &
                (predictions_gene['Case'] < (1-th))
            )
            P = np.where(predictions_gene['Case'] > th)
            N = np.where(predictions_gene['Case'] < (1-th))
            # Correct for testing split size
            no_gene += [((len(P[0])+len(N[0])) /
                         len(predictions_gene['True_Class']))*100]
        except: no_gene += [0]

    # Append Sampling percent to or_df
    or_df['SamplingPercent'] = no_gene

    # Calculate Histogram
    prob_case = predictions_gene[predictions_gene['True_Class'] == 1]
    prob_cont = predictions_gene[predictions_gene['True_Class'] == 0]
    _ = plt.hist(prob_case['Case'], bins=bins)
    _ = plt.hist(prob_cont['Case'], bins=bins)
    # Plot figure
    _, ax = plt.subplots(figsize = (10,10))
    plt.grid(False)
    if (xlim_lower != None) & (xlim_upper != None):
        bin_range = (xlim_lower, xlim_upper)
        ax.set_xlim(bin_range)
    else:
        bin_range = (
            pd.concat([prob_case['Case'], prob_cont['Case']], ignore_index = True).min(),
            pd.concat([prob_case['Case'], prob_cont['Case']], ignore_index = True).max()
        )
        ax.set_xlim(bin_range)

    ax.hist(prob_case['Case'], bins = bins, range = bin_range, color = 'red', alpha = 0.5, label = 'Cases')
    ax.hist(prob_cont['Case'], bins = bins, range = bin_range, color = 'blue', alpha = 0.5, label = 'Controls')
    if ylim[0] != None and ylim[1] != None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("Probability of being Case", size = 16)
    ax.set_ylabel("# of Samples", size = 16)
    plt.legend(loc = 'upper left', frameon = False, labelspacing = 0.01)
    ax2 = ax.twinx()
    plt.grid(False)
    ax2.plot(thrlist, or_df.OR, color = 'orange')
    ax2.set_ylabel("OR", color = 'orange', fontsize = 16)
    if ylim_or[0] != None and ylim_or[1] != None:
        ax2.set_ylim(ylim_or[0], ylim_or[1])
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    image = Image.open(buffer)

    # Get the KS Test p-value
    cases = predictions_gene[predictions_gene['True_Class'] == 1]
    controls = predictions_gene[predictions_gene['True_Class'] == 0]
    _, ks_p = ks_2samp(cases['Case'], controls['Case'], alternative = 'two-sided')

    return image, or_df, ks_p

def risk_prediction(feature_matrix: pd.DataFrame, train_samples: pd.DataFrame, test_samples: pd.DataFrame, models: Any = 'RF', rfe:bool = False, rfe_min_feature_ratio:float = 0.5, cores:int = 1, savepath: str = None) -> (Image, Image, pd.DataFrame): # type: ignore
    """
    Perform risk prediction using machine learning models.

    Args:
        feature_matrix (pd.DataFrame): The feature matrix containing the input features.
        train_samples (pd.DataFrame): The training samples.
        test_samples (pd.DataFrame): The testing samples.
        models (Any, optional): The machine learning models to use. Defaults to 'RF'.
        rfe (bool, optional): Whether to perform Recursive Feature Elimination (RFE). Defaults to False.
        rfe_min_feature_ratio (float, optional): The minimum feature ratio for RFE. Defaults to 0.5.
        cores (int, optional): The number of CPU cores to use. Defaults to 1.
        savepath (str, optional): The path to save the results. Defaults to None.

    Returns:
        eval_dist_plot (pd.DataFrame): The evaluation distribution plot.
        val_auroc_curve (pd.DataFrame): The validation AUROC curve.
        predictions_gene_df (pd.DataFrame): The predicted gene dataframe.
    """
    # Ensure feature matrix and train/test samples are in correct format
    train_samples = _validate_sample_dfs(train_samples, "Training")
    test_samples = _validate_sample_dfs(test_samples, "Testing")
    _validate_feature_matrix(feature_matrix, train_samples, test_samples)
    # Define the training and testing variables
    x_train, y_train = _generate_x_y_matrices(feature_matrix, train_samples)
    x_test, y_test = _generate_x_y_matrices(feature_matrix, test_samples)

    # Optimize the hyperparameters
    final_estimator, best_model, selected_features, best_optimization_df, rfe_importances, rfe_results, rfe_results_plot, bayes_results, bayes_results_plot, tested_models =  _optimize_features_hyperparameters(models, rfe, x_train, y_train, rfe_min_feature_ratio, cores)

    ## Get intra-sample model performance metrics
    # Cross-Validation
    cv_score = cross_val_score(final_estimator, x_train, y_train, cv=10, scoring='roc_auc')
    # Get Feature Weights
    feature_weights = _save_feature_weights(final_estimator, models, x_train)

    ## Test the final model
    x_test = x_test.loc[:,selected_features]
    y_pred = final_estimator.predict(x_test)
    acc_score = balanced_accuracy_score(y_test, y_pred)
    predictions_gene = final_estimator.predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, predictions_gene[:, 1])
    rocauc_score = auc(fpr, tpr)
    auc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    val_auroc_curve = _plot_auroc(fpr, tpr, rocauc_score)

    # Evaluate the true state v. predicted state
    predictions_gene_df = pd.DataFrame(predictions_gene, columns = ['Control', 'Case'])
    predictions_gene_df['True_Class'] = list(y_test)
    predictions_gene_df.index = x_test.index
    eval_dist_plot, or_df, ks_p = _plot_histo_w_or(predictions_gene_df, bins = 50)

    # Save the feature matrix
    if savepath:
        # Clean Savepath
        savepath = _fix_savepath(savepath)
        new_savepath = os.path.join(savepath, 'RiskPrediction/')
        os.makedirs(new_savepath, exist_ok=True)

        # Save the bayesian optimization results
        best_optimization_df.to_csv(new_savepath + 'best_optimization_results.csv', index=False)

        # Save training and testing intermediate split files
        input_file_savepath = new_savepath + 'IntermediateFiles/'
        os.makedirs(input_file_savepath, exist_ok=True)
        for df, name in zip([x_train, y_train, x_test, y_test], ['X_train', 'y_train', 'X_test', 'y_test']):
            df.to_csv(input_file_savepath + name + '.csv', index=False)

        # Save final files according to models tested
        for model in tested_models:
            # Create model-specific savepath
            model_savepath = new_savepath + model + '/'
            os.makedirs(model_savepath, exist_ok=True)
            # Save the rfe results
            rfe_savepath = model_savepath + 'RFE/'
            os.makedirs(rfe_savepath, exist_ok=True)
            if rfe_results[model] is not None:
                rfe_results[model].to_csv(rfe_savepath + 'rfe_results.csv', index=False)
            if rfe_importances[model] is not None:
                rfe_importances[model].to_csv(rfe_savepath + 'rfe_importances.csv', index=False)
            if rfe_results_plot[model] is not None:
                rfe_results_plot[model].save(rfe_savepath + 'rfe_results_plot.png')
            # Save the bayesian results
            bayes_savepath = model_savepath + 'BayesianOptimization/'
            os.makedirs(bayes_savepath, exist_ok=True)
            bayes_results[model].to_csv(bayes_savepath + 'bayes_optimization_results.csv', index=False)
            bayes_results_plot[model].save(bayes_savepath + 'bayes_optimization_results_plot.png')
            # Save the best model results
            if model == best_model:
                testing_savepath = model_savepath + 'TestingSamples/'
                os.makedirs(testing_savepath, exist_ok=True)
                # Save the feature weights
                if feature_weights is not None:
                    feature_weights.to_csv(testing_savepath + 'feature_weights.csv', index=True)
                # Save the final model
                with open(testing_savepath + 'final_model.pkl', 'wb') as f:
                    joblib.dump(final_estimator, f)
                # Save the final model predictions
                predictions_gene_df.to_csv(testing_savepath + 'predictions_gene.csv', index=True)
                auc_df.to_csv(testing_savepath + 'auc_df.csv', index=False)
                # Save the final model evaluation
                or_df.to_csv(testing_savepath + 'or_df.csv', index=False)
                eval_dist_plot.save(testing_savepath + 'eval_distribution_plot.png')
                val_auroc_curve.save(testing_savepath + 'eval_auroc_curve.png')
                # Save the final model performance metrics
                with open(testing_savepath + 'model_performance_metrics.txt', 'w') as f:
                    f.write(f'Intra-Training Sample 10x Cross-Validation ROC-AUC: {cv_score.mean()}\n')
                    f.write("\n")
                    f.write(f'Left-Out Sample Balanced Accuracy Score: {acc_score}\n')
                    f.write(f'Left-Out Sample ROC-AUC Score: {rocauc_score}\n')
                    f.write(f'Left-Out KS Test p-value b/w the two distributions: {ks_p}')

    return eval_dist_plot, val_auroc_curve, predictions_gene_df
#endregion


#region odds_ratios


def _run_exact_test(vcf_path:str):
    x =1 

def odds_ratios(vcf_path: str, genes:list, samples: pd.DataFrame, cores: int, savepath: str = None):


    if savepath:
        do_something = 1

#endregion


#region ea_distributions


def ea_distributions(variants: pd.DataFrame, min_af: float = 0.0, max_af: float = 1.0, ea_lower:int = 0, ea_upper: int = 100, savepath: str = "./") -> (Image): #type: ignore
    x = 1
    # Define the training and testing variables
#endregion
