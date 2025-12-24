"""
RNA-seq Drug Screening Dataset

Gene expression data from drug screening experiments testing
ADCs (antibody-drug conjugates) and free cytotoxic payloads.

Adapted to use workspace utilities.
"""

import sys
from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Workspace import setup
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.animation_utils import animator
from utils.debug_utils import quick_debug_setup

logger, config = quick_debug_setup(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "cache"

# Output directories for organized analysis steps
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPLORATORY_DIR = OUTPUT_DIR / "exploratory"
POPULATION_ANALYSIS_DIR = OUTPUT_DIR / "population_analysis"
DIFFERENTIAL_EXPRESSION_DIR = POPULATION_ANALYSIS_DIR / "differential_expression"
FEATURE_IMPORTANCE_DIR = POPULATION_ANALYSIS_DIR / "feature_importance"
STATISTICAL_TESTS_DIR = POPULATION_ANALYSIS_DIR / "statistical_tests"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
PATHWAY_ENRICHMENT_DIR = OUTPUT_DIR / "pathway_enrichment"


def _cached(func):
    """
    Decorator that caches DataFrame outputs as parquet files.
    
    Uses workspace file utilities for consistency.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{func.__name__}.parquet"

        # Always execute the function (Hamilton manages the DAG)
        result = func(*args, **kwargs)

        # Cache if it's a DataFrame
        if isinstance(result, pd.DataFrame):
            result.to_parquet(cache_path, index=False)
            logger.debug(f"Cached {func.__name__} to {cache_path}")

        return result

    # Preserve function metadata for Hamilton
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__
    return wrapper


@_cached
def sample_metadata() -> pd.DataFrame:
    """
    Load sample metadata for drug screening experiment.
    
    Returns a DataFrame with 52 samples across 3 plates (NEC03 P1, NEC03 P2, OD48 P3)
    testing various compounds at different concentrations:
    
    Compounds tested:
    - ADCs: Enhertu, Trodelvy, PADCEV, NTX1105Exatecan, NTX1105MMAE, 9B5
    - Free drugs: Free exatecan, Free MMAE
    - Controls: Ctrl, Dex w Cyt (various combinations)
    
    Columns:
    - sample_id: Sample identifier (TD009_X format)
    - plate: Plate identifier (NEC03 P1, NEC03 P2, OD48 P3)
    - compound: Drug/compound name
    - concentration: Drug concentration (nM)
    - well_ids: Plate well positions
    """
    data_path = PROJECT_ROOT / "input" / "Samples ID.xlsx"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Sample metadata file not found: {data_path}\n"
            "Please copy Samples ID.xlsx to input/ directory"
        )
    
    animator.show_info(f"Loading sample metadata from {data_path.name}")
    df = pd.read_excel(data_path, skiprows=2)

    # Clean up column names
    df = df.rename(columns={
        'sample ID in Savyon': 'sample_id',
        'Screen WellID': 'well_ids',
        'Plate': 'plate',
        'Compound': 'compound',
        'Conc': 'concentration'
    })

    # Keep only relevant columns
    df = df[['sample_id', 'plate', 'compound', 'concentration', 'well_ids']].copy()

    # Drop rows without sample_id
    df = df.dropna(subset=['sample_id'])

    # Forward fill plate info
    df['plate'] = df['plate'].ffill()

    # Clean compound names (strip whitespace)
    df['compound'] = df['compound'].str.strip()

    # Normalize sample IDs (some use . instead of _ for replicates)
    df['sample_id'] = df['sample_id'].str.replace('.', '_', regex=False)

    logger.info(f"Loaded sample metadata: {len(df)} samples")
    return df


@_cached
def raw_gene_counts() -> pd.DataFrame:
    """
    Load raw Salmon gene counts from RNA-seq experiment.
    
    Returns a DataFrame with 78,932 genes x 52 samples.
    Counts are from Salmon quantification (transcript-level, merged to gene).
    
    Structure:
    - gene_id: Ensembl gene ID (ENSG...)
    - gene_name: Gene symbol (e.g., TSPAN6, BRCA1)
    - TD009_X columns: Raw counts per sample
    
    Note: Counts may have decimal values due to Salmon's expectation-maximization
    algorithm for multi-mapped reads.
    """
    data_path = PROJECT_ROOT / "input" / "salmon_gene_counts.tsv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Gene counts file not found: {data_path}\n"
            "Please copy salmon_gene_counts.tsv to input/ directory"
        )
    
    animator.show_info(f"Loading gene counts from {data_path.name}")
    df = pd.read_csv(data_path, sep='\t')
    logger.info(f"Loaded gene counts: {df.shape[0]:,} genes x {df.shape[1]} columns")
    return df


@_cached
def gene_expression_matrix(raw_gene_counts: pd.DataFrame, sample_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare gene expression matrix with samples as columns and genes as rows.
    
    Matches sample IDs from metadata to gene count columns.
    Returns expression matrix ready for analysis (genes x samples).
    """
    animator.show_info("Preparing gene expression matrix...")
    
    # Get sample columns (TD009_X format)
    sample_cols = [col for col in raw_gene_counts.columns if col.startswith('TD009_')]
    
    # Create expression matrix (genes x samples)
    expr_matrix = raw_gene_counts[sample_cols].copy()
    expr_matrix.index = raw_gene_counts['gene_id']
    
    # Match sample IDs from metadata (both already have TD009_ prefix)
    metadata_samples = set(sample_metadata['sample_id'].astype(str))
    
    # Filter to samples that exist in both (keep TD009_ prefix)
    valid_samples = [col for col in sample_cols if col in metadata_samples]
    expr_matrix = expr_matrix[valid_samples]
    
    logger.info(f"Expression matrix: {expr_matrix.shape[0]:,} genes x {expr_matrix.shape[1]} samples")
    return expr_matrix


@_cached
def log_transformed_expression(gene_expression_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform gene expression (log2(count + 1)).
    
    Standard transformation for RNA-seq data to reduce variance and normalize.
    """
    animator.show_info("Log-transforming gene expression...")
    
    log_expr = np.log2(gene_expression_matrix + 1)
    logger.info(f"Log-transformed expression: {log_expr.shape[0]:,} genes x {log_expr.shape[1]} samples")
    return log_expr


@_cached
def pca_results(log_transformed_expression: pd.DataFrame, sample_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Perform PCA on log-transformed gene expression.
    
    Returns PCA results with sample metadata for visualization.
    """
    animator.show_info("Performing PCA analysis...")
    
    # Prepare data (transpose: samples x genes)
    expr_data = log_transformed_expression.T
    
    # Standardize
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_data)
    
    # PCA
    pca = PCA(n_components=min(10, expr_scaled.shape[0], expr_scaled.shape[1]))
    pca_result = pca.fit_transform(expr_scaled)
    
    # Create results DataFrame
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=expr_data.index
    )
    
    # Add sample metadata (sample_id already has TD009_ prefix in metadata)
    metadata_dict = sample_metadata.set_index('sample_id').to_dict('index')
    
    pca_df['sample_id'] = pca_df.index
    pca_df['compound'] = pca_df['sample_id'].map(
        lambda x: metadata_dict.get(x, {}).get('compound', 'Unknown') if x in metadata_dict else 'Unknown'
    )
    pca_df['concentration'] = pca_df['sample_id'].map(
        lambda x: metadata_dict.get(x, {}).get('concentration', 0) if x in metadata_dict else 0
    )
    pca_df['plate'] = pca_df['sample_id'].map(
        lambda x: metadata_dict.get(x, {}).get('plate', 'Unknown') if x in metadata_dict else 'Unknown'
    )
    
    # Add variance explained
    variance_explained = pca.explained_variance_ratio_
    logger.info(f"PCA complete: {len(variance_explained)} components, "
                f"PC1: {variance_explained[0]:.1%}, PC2: {variance_explained[1]:.1%}")
    
    return pca_df


@_cached
def sample_correlation_matrix(log_transformed_expression: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix between samples.
    
    Shows how similar samples are in terms of gene expression patterns.
    Correlation is computed across genes (samples x samples matrix).
    """
    animator.show_info("Calculating sample correlation matrix...")
    
    # Transpose: samples x genes (each row is a sample, each column is a gene)
    expr_data = log_transformed_expression.T
    
    # Calculate correlation between samples (across genes)
    # This gives us a samples x samples correlation matrix
    corr_matrix = expr_data.T.corr()  # Correlate samples (rows) with each other
    
    logger.info(f"Correlation matrix: {corr_matrix.shape[0]} samples x {corr_matrix.shape[1]} samples")
    return corr_matrix


@_cached
def compound_expression_summary(log_transformed_expression: pd.DataFrame, sample_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean expression per compound across all genes.
    
    Provides summary statistics for each compound.
    """
    animator.show_info("Calculating compound expression summaries...")
    
    # Match samples to compounds
    expr_data = log_transformed_expression.T.copy()
    expr_data['sample_id'] = expr_data.index  # Keep full TD009_X format
    
    # Merge with metadata (sample_id already has TD009_ prefix)
    metadata_lookup = sample_metadata.set_index('sample_id')[['compound', 'concentration']].to_dict('index')
    expr_data['compound'] = expr_data['sample_id'].map(
        lambda x: metadata_lookup.get(x, {}).get('compound', 'Unknown') if x in metadata_lookup else 'Unknown'
    )
    
    # Get only numeric columns (gene expression columns)
    numeric_cols = expr_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group by compound and calculate mean on numeric columns only
    compound_summary = expr_data.groupby('compound')[numeric_cols].mean()
    
    logger.info(f"Compound summary: {compound_summary.shape[0]} compounds x {compound_summary.shape[1]} genes")
    return compound_summary


def create_pca_plot(pca_results: pd.DataFrame) -> Path:
    """
    Create PCA plot colored by compound.
    
    Saves plot to output/exploratory/ directory.
    """
    animator.show_info("Creating PCA plot...")
    
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: PC1 vs PC2 colored by compound
    ax1 = axes[0]
    compounds = pca_results['compound'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(compounds)))
    color_map = dict(zip(compounds, colors))
    
    for compound in compounds:
        mask = pca_results['compound'] == compound
        ax1.scatter(
            pca_results.loc[mask, 'PC1'],
            pca_results.loc[mask, 'PC2'],
            label=compound,
            alpha=0.7,
            s=100,
            c=[color_map[compound]]
        )
    
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('PCA: PC1 vs PC2 (colored by compound)', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: PC1 vs PC2 colored by concentration
    ax2 = axes[1]
    scatter = ax2.scatter(
        pca_results['PC1'],
        pca_results['PC2'],
        c=pca_results['concentration'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('PCA: PC1 vs PC2 (colored by concentration)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Concentration (nM)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = EXPLORATORY_DIR / "pca_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved PCA plot", str(output_path))
    return output_path


def create_correlation_heatmap(sample_correlation_matrix: pd.DataFrame, sample_metadata: pd.DataFrame) -> Path:
    """
    Create correlation heatmap between samples.
    
    Saves plot to output/exploratory/ directory.
    """
    animator.show_info("Creating correlation heatmap...")
    
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare labels with compound info (sample_id already has TD009_ prefix)
    metadata_dict = sample_metadata.set_index('sample_id').to_dict('index')
    
    labels = []
    for sid in sample_correlation_matrix.index:
        if sid in metadata_dict:
            compound = metadata_dict[sid].get('compound', 'Unknown')
            conc = metadata_dict[sid].get('concentration', 0)
            labels.append(f"{sid}\n{compound}\n{conc}nM")
        else:
            labels.append(sid)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(
        sample_correlation_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Sample Correlation Matrix (Gene Expression)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Sample', fontsize=12)
    
    # Rotate labels for readability
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    
    plt.tight_layout()
    
    output_path = EXPLORATORY_DIR / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved correlation heatmap", str(output_path))
    return output_path


def create_compound_comparison_plot(compound_expression_summary: pd.DataFrame) -> Path:
    """
    Create comparison plot showing expression differences between compounds.
    
    Uses top variable genes across compounds.
    Saves to output/exploratory/ directory.
    """
    animator.show_info("Creating compound comparison plot...")
    
    EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate variance across compounds for each gene
    gene_variance = compound_expression_summary.var(axis=0).sort_values(ascending=False)
    
    # Get top 50 most variable genes
    top_genes = gene_variance.head(50).index
    
    # Prepare data for heatmap
    plot_data = compound_expression_summary[top_genes].T
    
    fig, ax = plt.subplots(figsize=(12, 14))
    
    sns.heatmap(
        plot_data,
        cmap='RdYlBu_r',
        center=plot_data.mean().mean(),
        cbar_kws={'label': 'Log2 Expression'},
        ax=ax,
        xticklabels=True,
        yticklabels=True
    )
    
    ax.set_title('Top 50 Most Variable Genes Across Compounds', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Compound', fontsize=12)
    ax.set_ylabel('Gene', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    
    plt.tight_layout()
    
    output_path = EXPLORATORY_DIR / "compound_comparison_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved compound comparison plot", str(output_path))
    return output_path


@_cached
def create_all_exploratory_plots(
    pca_results: pd.DataFrame,
    sample_correlation_matrix: pd.DataFrame,
    compound_expression_summary: pd.DataFrame,
    sample_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Create all exploratory visualizations.
    
    This is a convenience function that creates all plots.
    Returns a summary DataFrame of created plots.
    """
    animator.show_operation_start("Creating Exploratory Visualizations", "Generating PCA, correlation, and comparison plots")
    
    plots_created = []
    
    # Create PCA plot
    pca_path = create_pca_plot(pca_results)
    plots_created.append({'plot_type': 'PCA', 'path': str(pca_path)})
    
    # Create correlation heatmap
    corr_path = create_correlation_heatmap(sample_correlation_matrix, sample_metadata)
    plots_created.append({'plot_type': 'Correlation', 'path': str(corr_path)})
    
    # Create compound comparison
    comp_path = create_compound_comparison_plot(compound_expression_summary)
    plots_created.append({'plot_type': 'Compound Comparison', 'path': str(comp_path)})
    
    plots_df = pd.DataFrame(plots_created)
    
    animator.show_operation_complete("Exploratory Visualizations", 0)
    animator.show_info(f"Created {len(plots_created)} visualization plots")
    
    return plots_df


# =============================================================================
# Population Analysis - Understanding What Separates the 2 Populations
# =============================================================================


@_cached
def population_assignment(
    pca_results: pd.DataFrame,
    sample_metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign samples to Population 1 or Population 2 based on PCA clustering.
    
    Population 1: Controls and Dex w Cyt compounds (negative PC1, low expression)
    Population 2: ADCs and free drugs (positive PC1, high expression)
    
    Returns DataFrame with sample_id and population assignment.
    """
    animator.show_info("Assigning samples to populations based on PCA...")
    
    # Define population based on PC1 (main separation axis)
    # Negative PC1 = Population 1 (controls/Dex w Cyt)
    # Positive PC1 = Population 2 (ADCs/free drugs)
    
    assignment = pca_results[['sample_id', 'compound', 'PC1', 'PC2']].copy()
    assignment['population'] = assignment['PC1'].apply(
        lambda x: 'Population_1' if x < 0 else 'Population_2'
    )
    
    # Count samples per population
    pop_counts = assignment['population'].value_counts()
    logger.info(f"Population assignment: {pop_counts['Population_1']} samples in Pop1, "
                f"{pop_counts['Population_2']} samples in Pop2")
    
    return assignment


@_cached
def population_compound_summary(
    population_assignment: pd.DataFrame
) -> pd.DataFrame:
    """
    Summary of compounds in each population.
    
    Shows which compounds belong to which population and their counts.
    """
    summary = population_assignment.groupby(['population', 'compound']).size().reset_index(name='count')
    summary = summary.sort_values(['population', 'count'], ascending=[True, False])
    
    logger.info(f"Population compound summary: {len(summary)} compound-population combinations")
    return summary


@_cached
def differential_expression_populations(
    log_transformed_expression: pd.DataFrame,
    population_assignment: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate differential expression between Population 1 and Population 2.
    
    For each gene, calculates:
    - Mean expression in Pop1 vs Pop2
    - Fold change (Pop2 / Pop1)
    - Log2 fold change
    - Absolute difference
    
    Returns ranked list of genes that best separate the populations.
    """
    animator.show_info("Calculating differential expression between populations...")
    
    # Get expression matrix (genes x samples)
    expr_data = log_transformed_expression.T.copy()
    expr_data['sample_id'] = expr_data.index
    
    # Merge with population assignment
    pop_lookup = population_assignment.set_index('sample_id')['population'].to_dict()
    expr_data['population'] = expr_data['sample_id'].map(
        lambda x: pop_lookup.get(x, 'Unknown')
    )
    
    # Filter to known populations
    expr_data = expr_data[expr_data['population'].isin(['Population_1', 'Population_2'])]
    
    # Get numeric columns (genes)
    gene_cols = expr_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate mean expression per population
    pop1_means = expr_data[expr_data['population'] == 'Population_1'][gene_cols].mean()
    pop2_means = expr_data[expr_data['population'] == 'Population_2'][gene_cols].mean()
    
    # Calculate fold change and differences
    diff_results = pd.DataFrame({
        'gene_id': gene_cols,
        'pop1_mean': pop1_means.values,
        'pop2_mean': pop2_means.values,
        'mean_difference': (pop2_means - pop1_means).values,
        'fold_change': (pop2_means / (pop1_means + 1e-10)).values,  # Add small value to avoid div by zero
        'log2_fold_change': (np.log2(pop2_means + 1) - np.log2(pop1_means + 1)).values,
        'abs_difference': np.abs(pop2_means - pop1_means).values
    })
    
    # Rank by absolute difference (genes that separate populations best)
    diff_results = diff_results.sort_values('abs_difference', ascending=False)
    
    logger.info(f"Differential expression: {len(diff_results)} genes analyzed")
    logger.info(f"Top separating gene: {diff_results.iloc[0]['gene_id']} "
                f"(abs diff: {diff_results.iloc[0]['abs_difference']:.2f})")
    
    return diff_results


@_cached
def top_separating_genes(
    differential_expression_populations: pd.DataFrame,
    raw_gene_counts: pd.DataFrame,
    n_genes: int = 50
) -> pd.DataFrame:
    """
    Get top N genes that best separate the two populations.
    
    Includes gene names for biological interpretation.
    """
    animator.show_info(f"Identifying top {n_genes} separating genes...")
    
    # Get top genes by absolute difference
    top_genes = differential_expression_populations.head(n_genes).copy()
    
    # Add gene names
    gene_name_map = raw_gene_counts.set_index('gene_id')['gene_name'].to_dict()
    top_genes['gene_name'] = top_genes['gene_id'].map(
        lambda x: gene_name_map.get(x, 'Unknown')
    )
    
    logger.info(f"Top {n_genes} separating genes identified")
    return top_genes


@_cached
def population_statistical_tests(
    log_transformed_expression: pd.DataFrame,
    population_assignment: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform statistical tests to validate population differences.
    
    For each gene, performs:
    - T-test (Population 1 vs Population 2)
    - Mann-Whitney U test (non-parametric)
    - Effect size (Cohen's d)
    
    Returns genes with significant differences (p < 0.05).
    """
    from scipy import stats
    
    animator.show_info("Performing statistical tests between populations...")
    
    # Get expression matrix
    expr_data = log_transformed_expression.T.copy()
    expr_data['sample_id'] = expr_data.index
    
    # Merge with population assignment
    pop_lookup = population_assignment.set_index('sample_id')['population'].to_dict()
    expr_data['population'] = expr_data['sample_id'].map(
        lambda x: pop_lookup.get(x, 'Unknown')
    )
    
    # Filter to known populations
    expr_data = expr_data[expr_data['population'].isin(['Population_1', 'Population_2'])]
    
    # Get numeric columns (genes) - limit to top variable genes for efficiency
    gene_cols = expr_data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to top 1000 most variable genes for computational efficiency
    gene_variance = expr_data[gene_cols].var().sort_values(ascending=False)
    top_var_genes = gene_variance.head(1000).index.tolist()
    
    results = []
    
    with animator.progress_bar(len(top_var_genes), "Testing genes", "genes") as progress:
        for gene in top_var_genes:
            pop1_values = expr_data[expr_data['population'] == 'Population_1'][gene].values
            pop2_values = expr_data[expr_data['population'] == 'Population_2'][gene].values
            
            # Remove any NaN values
            pop1_values = pop1_values[~np.isnan(pop1_values)]
            pop2_values = pop2_values[~np.isnan(pop2_values)]
            
            if len(pop1_values) > 2 and len(pop2_values) > 2:
                # T-test
                t_stat, t_pvalue = stats.ttest_ind(pop2_values, pop1_values)
                
                # Mann-Whitney U test
                u_stat, u_pvalue = stats.mannwhitneyu(pop2_values, pop1_values, alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(pop1_values) - 1) * np.var(pop1_values) + 
                                     (len(pop2_values) - 1) * np.var(pop2_values)) / 
                                    (len(pop1_values) + len(pop2_values) - 2))
                cohens_d = (np.mean(pop2_values) - np.mean(pop1_values)) / (pooled_std + 1e-10)
                
                results.append({
                    'gene_id': gene,
                    't_statistic': t_stat,
                    't_pvalue': t_pvalue,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pvalue,
                    'cohens_d': cohens_d,
                    'pop1_mean': np.mean(pop1_values),
                    'pop2_mean': np.mean(pop2_values),
                    'significant_t_test': t_pvalue < 0.05,
                    'significant_u_test': u_pvalue < 0.05
                })
            
            animator.step_progress(progress, 1, f"Tested {gene}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('t_pvalue')
    
    n_sig = results_df['significant_t_test'].sum()
    logger.info(f"Statistical tests: {n_sig} genes significantly different (p < 0.05) out of {len(results_df)} tested")
    
    return results_df


def create_population_separation_plot(
    top_separating_genes: pd.DataFrame,
    log_transformed_expression: pd.DataFrame,
    population_assignment: pd.DataFrame
) -> Path:
    """
    Create visualization showing how top separating genes distinguish populations.
    
    Shows expression of top genes in Population 1 vs Population 2.
    """
    animator.show_info("Creating population separation plot...")
    
    DIFFERENTIAL_EXPRESSION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get top 20 genes
    top_20 = top_separating_genes.head(20)
    
    # Get expression data
    expr_data = log_transformed_expression.T.copy()
    expr_data['sample_id'] = expr_data.index
    
    # Merge with population
    pop_lookup = population_assignment.set_index('sample_id')['population'].to_dict()
    expr_data['population'] = expr_data['sample_id'].map(
        lambda x: pop_lookup.get(x, 'Unknown')
    )
    expr_data = expr_data[expr_data['population'].isin(['Population_1', 'Population_2'])]
    
    # Prepare data for plotting
    plot_data = []
    for _, row in top_20.iterrows():
        gene_id = row['gene_id']
        gene_name = row.get('gene_name', gene_id)
        
        pop1_vals = expr_data[expr_data['population'] == 'Population_1'][gene_id].values
        pop2_vals = expr_data[expr_data['population'] == 'Population_2'][gene_id].values
        
        for val in pop1_vals:
            if not np.isnan(val):
                plot_data.append({'gene': f"{gene_name}\n({gene_id})", 'population': 'Population 1', 'expression': val})
        for val in pop2_vals:
            if not np.isnan(val):
                plot_data.append({'gene': f"{gene_name}\n({gene_id})", 'population': 'Population 2', 'expression': val})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Box plot
    sns.boxplot(data=plot_df, x='gene', y='expression', hue='population', ax=ax)
    ax.set_title('Top 20 Genes Separating Population 1 vs Population 2', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Gene', fontsize=12)
    ax.set_ylabel('Log2 Expression', fontsize=12)
    ax.legend(title='Population', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = DIFFERENTIAL_EXPRESSION_DIR / "population_separation_top_genes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved population separation plot", str(output_path))
    return output_path


def create_volcano_plot(
    population_statistical_tests: pd.DataFrame,
    raw_gene_counts: pd.DataFrame
) -> Path:
    """
    Create volcano plot showing statistical significance vs fold change.
    
    X-axis: Log2 fold change (Pop2 / Pop1)
    Y-axis: -log10(p-value)
    """
    animator.show_info("Creating volcano plot...")
    
    STATISTICAL_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate log2 fold change from means
    plot_data = population_statistical_tests.copy()
    plot_data['log2_fold_change'] = np.log2((plot_data['pop2_mean'] + 1) / (plot_data['pop1_mean'] + 1))
    plot_data['neg_log10_pvalue'] = -np.log10(plot_data['t_pvalue'] + 1e-10)  # Avoid log(0)
    
    # Add gene names
    gene_name_map = raw_gene_counts.set_index('gene_id')['gene_name'].to_dict()
    plot_data['gene_name'] = plot_data['gene_id'].map(
        lambda x: gene_name_map.get(x, 'Unknown')
    )
    
    # Mark significant genes
    plot_data['significant'] = plot_data['t_pvalue'] < 0.05
    plot_data['high_effect'] = np.abs(plot_data['cohens_d']) > 0.5
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot non-significant
    non_sig = plot_data[~plot_data['significant']]
    ax.scatter(non_sig['log2_fold_change'], non_sig['neg_log10_pvalue'], 
               alpha=0.3, c='gray', s=30, label='Not significant')
    
    # Plot significant
    sig = plot_data[plot_data['significant']]
    ax.scatter(sig['log2_fold_change'], sig['neg_log10_pvalue'], 
               alpha=0.7, c='red', s=50, label='Significant (p < 0.05)')
    
    # Highlight top genes
    top_genes = plot_data.nlargest(10, 'neg_log10_pvalue')
    for _, row in top_genes.iterrows():
        ax.annotate(row['gene_name'], 
                   (row['log2_fold_change'], row['neg_log10_pvalue']),
                   fontsize=8, alpha=0.8)
    
    # Add significance threshold line
    ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p = 0.05')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Log2 Fold Change (Population 2 / Population 1)', fontsize=12)
    ax.set_ylabel('-Log10(p-value)', fontsize=12)
    ax.set_title('Volcano Plot: Population 1 vs Population 2', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = STATISTICAL_TESTS_DIR / "volcano_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved volcano plot", str(output_path))
    return output_path


def create_feature_importance_plot(
    top_separating_genes: pd.DataFrame
) -> Path:
    """
    Create bar plot showing top genes ranked by separation power.
    """
    animator.show_info("Creating feature importance plot...")
    
    FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get top 30 genes
    top_30 = top_separating_genes.head(30).copy()
    
    # Create labels with gene names
    top_30['label'] = top_30.apply(
        lambda row: f"{row.get('gene_name', 'Unknown')}\n({row['gene_id']})" 
        if pd.notna(row.get('gene_name')) else row['gene_id'],
        axis=1
    )
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color by direction (up in Pop2 = red, down in Pop2 = blue)
    colors = ['red' if fc > 0 else 'blue' for fc in top_30['log2_fold_change']]
    
    bars = ax.barh(range(len(top_30)), top_30['abs_difference'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(top_30['label'], fontsize=8)
    ax.set_xlabel('Absolute Expression Difference (Pop2 - Pop1)', fontsize=12)
    ax.set_title('Top 30 Genes Separating Populations (Feature Importance)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()  # Top gene at top
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Higher in Population 2'),
        Patch(facecolor='blue', alpha=0.7, label='Higher in Population 1')
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    
    plt.tight_layout()
    
    output_path = FEATURE_IMPORTANCE_DIR / "feature_importance_top_genes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved feature importance plot", str(output_path))
    return output_path


def create_population_summary_plot(
    population_assignment: pd.DataFrame,
    pca_results: pd.DataFrame
) -> Path:
    """
    Create summary visualization showing population separation in PCA space.
    """
    animator.show_info("Creating population summary plot...")
    
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Merge PCA with population assignment
    plot_data = pca_results.merge(
        population_assignment[['sample_id', 'population']],
        on='sample_id',
        how='left'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: PCA colored by population
    ax1 = axes[0]
    pop1 = plot_data[plot_data['population'] == 'Population_1']
    pop2 = plot_data[plot_data['population'] == 'Population_2']
    
    ax1.scatter(pop1['PC1'], pop1['PC2'], alpha=0.7, s=100, 
               c='blue', label='Population 1 (Controls/Dex)', edgecolors='black', linewidths=0.5)
    ax1.scatter(pop2['PC1'], pop2['PC2'], alpha=0.7, s=100, 
               c='red', label='Population 2 (ADCs/Free drugs)', edgecolors='black', linewidths=0.5)
    
    ax1.set_xlabel('PC1 (25.0% variance)', fontsize=12)
    ax1.set_ylabel('PC2 (10.4% variance)', fontsize=12)
    ax1.set_title('PCA: Population Separation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Population distribution by compound
    ax2 = axes[1]
    pop_compound = plot_data.groupby(['population', 'compound']).size().unstack(fill_value=0)
    pop_compound.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
    ax2.set_xlabel('Population', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution: Population by Compound', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = VISUALIZATIONS_DIR / "population_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved population summary plot", str(output_path))
    return output_path


@_cached
def create_all_population_analysis_plots(
    top_separating_genes: pd.DataFrame,
    population_statistical_tests: pd.DataFrame,
    population_assignment: pd.DataFrame,
    pca_results: pd.DataFrame,
    log_transformed_expression: pd.DataFrame,
    raw_gene_counts: pd.DataFrame
) -> pd.DataFrame:
    """
    Create all population analysis visualizations.
    
    This is a convenience function that creates all plots for population analysis.
    Returns a summary DataFrame of created plots.
    """
    animator.show_operation_start("Creating Population Analysis Visualizations", 
                                  "Generating separation, volcano, and feature importance plots")
    
    plots_created = []
    
    # Create population separation plot
    sep_path = create_population_separation_plot(
        top_separating_genes, log_transformed_expression, population_assignment
    )
    plots_created.append({'plot_type': 'Population Separation', 'path': str(sep_path)})
    
    # Create volcano plot
    volc_path = create_volcano_plot(population_statistical_tests, raw_gene_counts)
    plots_created.append({'plot_type': 'Volcano Plot', 'path': str(volc_path)})
    
    # Create feature importance plot
    feat_path = create_feature_importance_plot(top_separating_genes)
    plots_created.append({'plot_type': 'Feature Importance', 'path': str(feat_path)})
    
    # Create population summary
    summ_path = create_population_summary_plot(population_assignment, pca_results)
    plots_created.append({'plot_type': 'Population Summary', 'path': str(summ_path)})
    
    plots_df = pd.DataFrame(plots_created)
    
    animator.show_operation_complete("Population Analysis Visualizations", 0)
    animator.show_info(f"Created {len(plots_created)} population analysis plots")
    
    return plots_df


# =============================================================================
# Pathway Enrichment Analysis
# =============================================================================


@_cached
def upregulated_genes_population2(
    top_separating_genes: pd.DataFrame,
    n_genes: int = 500
) -> pd.DataFrame:
    """
    Get top N upregulated genes in Population 2.
    
    These are genes that are higher in Population 2 (ADCs/Free drugs)
    compared to Population 1 (Controls/Dex).
    """
    animator.show_info(f"Identifying top {n_genes} upregulated genes in Population 2...")
    
    # Filter to genes upregulated in Pop2 (positive log2_fold_change)
    upregulated = top_separating_genes[top_separating_genes['log2_fold_change'] > 0].copy()
    upregulated = upregulated.sort_values('abs_difference', ascending=False)
    
    # Get top N
    top_upregulated = upregulated.head(n_genes)
    
    logger.info(f"Top {len(top_upregulated)} upregulated genes in Population 2 identified")
    return top_upregulated


@_cached
def pathway_enrichment_analysis(
    upregulated_genes_population2: pd.DataFrame,
    raw_gene_counts: pd.DataFrame
) -> pd.DataFrame:
    """
    Perform pathway enrichment analysis on upregulated genes.
    
    Uses gseapy to test for enrichment in:
    - GO (Gene Ontology) terms
    - KEGG pathways
    - Reactome pathways
    
    Returns enriched pathways with p-values and FDR.
    """
    try:
        import gseapy as gp
    except ImportError:
        animator.show_error("gseapy not installed. Install with: pip install gseapy")
        logger.error("gseapy not installed")
        return pd.DataFrame()
    
    animator.show_info("Performing pathway enrichment analysis...")
    
    PATHWAY_ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get gene names (convert Ensembl IDs to gene symbols if available)
    gene_list = upregulated_genes_population2['gene_id'].tolist()
    
    # Try to get gene symbols
    gene_name_map = raw_gene_counts.set_index('gene_id')['gene_name'].to_dict()
    gene_symbols = []
    for gene_id in gene_list:
        gene_name = gene_name_map.get(gene_id, gene_id)
        # Use gene name if available and not 'Unknown', otherwise use gene_id
        if gene_name and gene_name != 'Unknown' and pd.notna(gene_name):
            gene_symbols.append(str(gene_name))
        else:
            # Try to extract gene symbol from Ensembl ID if possible
            gene_symbols.append(str(gene_id))
    
    logger.info(f"Using {len(gene_symbols)} genes for enrichment (first 5: {gene_symbols[:5]})")
    
    # Run enrichment analysis
    try:
        # GO enrichment
        animator.show_info("Running GO enrichment...")
        go_enr = gp.enrichr(
            gene_list=gene_symbols,
            gene_sets=['GO_Biological_Process_2023', 'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023'],
            organism='human',
            outdir=None,
            no_plot=True,
            verbose=False
        )
        
        # KEGG enrichment
        animator.show_info("Running KEGG pathway enrichment...")
        kegg_enr = gp.enrichr(
            gene_list=gene_symbols,
            gene_sets=['KEGG_2021_Human'],
            organism='human',
            outdir=None,
            no_plot=True,
            verbose=False
        )
        
        # Reactome enrichment
        animator.show_info("Running Reactome pathway enrichment...")
        reactome_enr = gp.enrichr(
            gene_list=gene_symbols,
            gene_sets=['Reactome_2022'],
            organism='human',
            outdir=None,
            no_plot=True,
            verbose=False
        )
        
        # Combine results - gseapy returns results as a dictionary
        all_results = []
        
        # Helper to extract results from gseapy output
        def extract_results(enr_obj, source_name):
            if enr_obj is None:
                return
            # gseapy returns results in .res2d DataFrame
            if hasattr(enr_obj, 'res2d') and enr_obj.res2d is not None:
                results = enr_obj.res2d.copy()
                if len(results) > 0:
                    results['source'] = source_name
                    all_results.append(results)
            # Also check .results dictionary format
            elif hasattr(enr_obj, 'results'):
                for term_set, results in enr_obj.results.items():
                    if results is not None and isinstance(results, pd.DataFrame) and len(results) > 0:
                        results = results.copy()
                        results['source'] = f"{source_name}_{term_set}"
                        all_results.append(results)
        
        extract_results(go_enr, 'GO')
        extract_results(kegg_enr, 'KEGG')
        extract_results(reactome_enr, 'Reactome')
        
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            # Sort by adjusted p-value
            if 'Adjusted P-value' in combined_results.columns:
                combined_results = combined_results.sort_values('Adjusted P-value')
            
            # Filter significant (FDR < 0.05)
            significant = combined_results[combined_results['Adjusted P-value'] < 0.05]
            
            logger.info(f"Pathway enrichment: {len(significant)} significant pathways (FDR < 0.05) out of {len(combined_results)} total")
            
            return combined_results
        else:
            logger.warning("No enrichment results returned")
            return pd.DataFrame()
            
    except Exception as e:
        animator.show_error(f"Pathway enrichment failed: {str(e)}")
        logger.error(f"Pathway enrichment error: {e}", exc_info=True)
        return pd.DataFrame()


def create_pathway_enrichment_plot(
    pathway_enrichment_analysis: pd.DataFrame
) -> Path:
    """
    Create visualization of top enriched pathways.
    """
    animator.show_info("Creating pathway enrichment plot...")
    
    PATHWAY_ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    
    if len(pathway_enrichment_analysis) == 0:
        animator.show_warning("No pathway enrichment results to plot")
        return PATHWAY_ENRICHMENT_DIR / "pathway_enrichment_empty.png"
    
    # Get top 20 significant pathways
    significant = pathway_enrichment_analysis[pathway_enrichment_analysis['Adjusted P-value'] < 0.05]
    top_pathways = significant.head(20).copy()
    
    if len(top_pathways) == 0:
        animator.show_warning("No significant pathways to plot")
        return PATHWAY_ENRICHMENT_DIR / "pathway_enrichment_no_significant.png"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Calculate -log10(adjusted p-value)
    top_pathways['neg_log10_padj'] = -np.log10(top_pathways['Adjusted P-value'] + 1e-10)
    
    # Sort by significance
    top_pathways = top_pathways.sort_values('neg_log10_padj', ascending=True)
    
    # Create horizontal bar plot
    y_pos = range(len(top_pathways))
    bars = ax.barh(y_pos, top_pathways['neg_log10_padj'], 
                   color='steelblue', alpha=0.7)
    
    # Set labels
    ax.set_yticks(y_pos)
    # Truncate long pathway names
    labels = [name[:60] + '...' if len(name) > 60 else name 
              for name in top_pathways['Term']]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('-Log10(Adjusted P-value)', fontsize=12)
    ax.set_title('Top 20 Enriched Pathways (Population 2 Upregulated Genes)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add source information
    source_colors = {
        'GO_Biological_Process_2023': 'steelblue',
        'GO_Molecular_Function_2023': 'darkgreen',
        'GO_Cellular_Component_2023': 'purple',
        'KEGG_2021_Human': 'red',
        'Reactome_2022': 'orange'
    }
    
    # Color bars by source
    for i, (idx, row) in enumerate(top_pathways.iterrows()):
        source = row.get('source', '')
        color = source_colors.get(source, 'steelblue')
        bars[i].set_color(color)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=source.replace('_', ' ')) 
                      for source, color in source_colors.items()]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    
    plt.tight_layout()
    
    output_path = PATHWAY_ENRICHMENT_DIR / "pathway_enrichment_top_pathways.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved pathway enrichment plot", str(output_path))
    return output_path


@_cached
def create_all_pathway_analysis(
    pathway_enrichment_analysis: pd.DataFrame
) -> pd.DataFrame:
    """
    Create all pathway enrichment visualizations and save results.
    """
    animator.show_operation_start("Pathway Enrichment Analysis", 
                                  "Creating pathway enrichment visualizations")
    
    plots_created = []
    
    # Create pathway plot
    plot_path = create_pathway_enrichment_plot(pathway_enrichment_analysis)
    plots_created.append({'plot_type': 'Pathway Enrichment', 'path': str(plot_path)})
    
    # Save enrichment results to CSV
    if len(pathway_enrichment_analysis) > 0:
        csv_path = PATHWAY_ENRICHMENT_DIR / "pathway_enrichment_results.csv"
        pathway_enrichment_analysis.to_csv(csv_path, index=False)
        animator.show_file_operation("Saved pathway enrichment results", str(csv_path))
        
        # Save significant pathways separately
        significant = pathway_enrichment_analysis[pathway_enrichment_analysis['Adjusted P-value'] < 0.05]
        if len(significant) > 0:
            sig_path = PATHWAY_ENRICHMENT_DIR / "significant_pathways.csv"
            significant.to_csv(sig_path, index=False)
            animator.show_file_operation("Saved significant pathways", str(sig_path))
    
    plots_df = pd.DataFrame(plots_created)
    
    animator.show_operation_complete("Pathway Enrichment Analysis", 0)
    animator.show_info(f"Pathway enrichment analysis complete")
    
    return plots_df

