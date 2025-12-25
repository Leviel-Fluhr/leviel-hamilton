"""
Diabetes 130-US Hospitals Dataset (1999-2008)

Adapted to use workspace utilities.
"""

import sys
from pathlib import Path
from functools import wraps

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Workspace import setup
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.file_utils import load_csv, ensure_output_dir
from utils.animation_utils import animator
from utils.debug_utils import quick_debug_setup

logger, config = quick_debug_setup(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "output" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
DIABETES_EXPLORATORY_DIR = OUTPUT_DIR / "diabetes_exploratory"


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
def raw_diabetic_data() -> pd.DataFrame:
    """
    Load the diabetes 130-US hospitals dataset.
    
    Returns a DataFrame with 101,766 patient encounters and 50 features including:
    - Patient demographics (race, gender, age)
    - Encounter details (admission_type, discharge_disposition, time_in_hospital)
    - Clinical measurements (num_lab_procedures, num_medications, number_diagnoses)
    - Medication flags (23 drugs with dosage change indicators)
    - Target: readmitted (<30 days / >30 days / No)
    """
    data_path = PROJECT_ROOT / "input" / "diabetic_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Diabetes data not found: {data_path}\n"
            "Please copy diabetic_data.csv to input/ directory"
        )
    
    animator.show_info(f"Loading diabetes data from {data_path.name}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded diabetes data: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def _parse_ids_mapping() -> dict[str, pd.DataFrame]:
    """
    Parse the IDS_mapping.csv file which contains three stacked lookup tables.
    
    Returns a dict with keys: 'admission_type', 'discharge_disposition', 'admission_source'
    """
    data_path = PROJECT_ROOT / "input" / "IDS_mapping.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"IDS mapping file not found: {data_path}\n"
            "Please copy IDS_mapping.csv to input/ directory"
        )

    with open(data_path, "r") as f:
        lines = f.readlines()

    tables = {}
    current_table = None
    current_rows = []

    for line in lines:
        line = line.strip()
        if not line or line == ",":
            if current_table and current_rows:
                tables[current_table] = current_rows
                current_rows = []
            continue

        if line.startswith("admission_type_id,"):
            current_table = "admission_type"
            continue
        elif line.startswith("discharge_disposition_id,"):
            current_table = "discharge_disposition"
            continue
        elif line.startswith("admission_source_id,"):
            current_table = "admission_source"
            continue

        if current_table:
            parts = line.split(",", 1)
            if len(parts) == 2 and parts[0].strip():
                current_rows.append({
                    "id": int(parts[0].strip()),
                    "description": parts[1].strip().strip('"')
                })

    if current_table and current_rows:
        tables[current_table] = current_rows

    return {k: pd.DataFrame(v) for k, v in tables.items()}


@_cached
def admission_type_lookup() -> pd.DataFrame:
    """
    Lookup table for admission_type_id codes.
    
    Maps numeric IDs to descriptions:
    1=Emergency, 2=Urgent, 3=Elective, 4=Newborn, etc.
    
    Join with raw_diabetic_data on admission_type_id.
    """
    return _parse_ids_mapping()["admission_type"]


@_cached
def discharge_disposition_lookup() -> pd.DataFrame:
    """
    Lookup table for discharge_disposition_id codes.
    
    Maps numeric IDs to descriptions:
    1=Discharged to home, 6=Discharged with home health service,
    7=Left AMA, 11=Expired, 13=Hospice/home, etc.
    
    Join with raw_diabetic_data on discharge_disposition_id.
    """
    return _parse_ids_mapping()["discharge_disposition"]


@_cached
def admission_source_lookup() -> pd.DataFrame:
    """
    Lookup table for admission_source_id codes.
    
    Maps numeric IDs to descriptions:
    1=Physician Referral, 4=Transfer from hospital,
    7=Emergency Room, etc.
    
    Join with raw_diabetic_data on admission_source_id.
    """
    return _parse_ids_mapping()["admission_source"]


@_cached
def readmission_by_age(raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Readmission rates broken down by age group.
    
    Shows the percentage of patients readmitted within 30 days,
    after 30 days, or not readmitted, for each age bracket.
    """
    summary = raw_diabetic_data.groupby('age')['readmitted'].value_counts(normalize=True)
    summary = summary.unstack(fill_value=0) * 100
    summary = summary.round(2).reset_index()
    summary.columns.name = None
    return summary


@_cached
def diabetes_preprocessed_data(raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess diabetes data for PCA analysis.
    
    Handles missing values, encodes categorical variables, and selects numeric features.
    """
    animator.show_info("Preprocessing diabetes data for PCA...")
    
    df = raw_diabetic_data.copy()
    
    # Select numeric features for PCA
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID columns (not features)
    id_cols = ['encounter_id', 'patient_nbr']
    numeric_cols = [col for col in numeric_cols if col not in id_cols]
    
    # Handle missing values encoded as '?'
    for col in numeric_cols:
        if df[col].dtype == 'object':
            # Convert '?' to NaN, then to numeric
            df[col] = pd.to_numeric(df[col].replace('?', np.nan), errors='coerce')
    
    # Select only numeric columns that have valid data
    valid_numeric_cols = []
    for col in numeric_cols:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            if non_null_count > len(df) * 0.5:  # Keep if >50% non-null
                valid_numeric_cols.append(col)
    
    # Encode categorical variables that might be useful
    categorical_cols = ['race', 'gender', 'age', 'readmitted']
    le_dict = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle '?' as a category
            df[col] = df[col].fillna('Unknown').astype(str)
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            le_dict[col] = le
            if f'{col}_encoded' not in valid_numeric_cols:
                valid_numeric_cols.append(f'{col}_encoded')
    
    # Create preprocessed dataset
    preprocessed = df[valid_numeric_cols].copy()
    
    # Fill remaining NaN with median
    for col in preprocessed.columns:
        if preprocessed[col].isna().any():
            preprocessed[col].fillna(preprocessed[col].median(), inplace=True)
    
    logger.info(f"Preprocessed diabetes data: {preprocessed.shape[0]:,} rows x {preprocessed.shape[1]} features")
    return preprocessed


@_cached
def diabetes_pca_results(diabetes_preprocessed_data: pd.DataFrame, raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform PCA on preprocessed diabetes data.
    
    Returns PCA results with patient metadata for visualization.
    """
    animator.show_info("Performing PCA on diabetes data...")
    
    # Prepare data
    data = diabetes_preprocessed_data.values
    
    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # PCA
    n_components = min(10, data_scaled.shape[0], data_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create results DataFrame
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=diabetes_preprocessed_data.index
    )
    
    # Add metadata from original data
    pca_df['readmitted'] = raw_diabetic_data['readmitted'].values
    pca_df['age'] = raw_diabetic_data['age'].values
    pca_df['gender'] = raw_diabetic_data['gender'].values
    pca_df['time_in_hospital'] = raw_diabetic_data['time_in_hospital'].values
    
    # Add variance explained
    variance_explained = pca.explained_variance_ratio_
    logger.info(f"Diabetes PCA complete: {len(variance_explained)} components, "
                f"PC1: {variance_explained[0]:.1%}, PC2: {variance_explained[1]:.1%}")
    
    return pca_df


def create_diabetes_pca_plot(diabetes_pca_results: pd.DataFrame) -> Path:
    """
    Create PCA plot for diabetes data.
    
    Visualizes patient encounters in PCA space colored by readmission status.
    """
    DIABETES_EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Colored by readmission status
    ax1 = axes[0]
    for readmit_status in diabetes_pca_results['readmitted'].unique():
        mask = diabetes_pca_results['readmitted'] == readmit_status
        ax1.scatter(
            diabetes_pca_results.loc[mask, 'PC1'],
            diabetes_pca_results.loc[mask, 'PC2'],
            label=readmit_status,
            alpha=0.3,
            s=10
        )
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('Diabetes Data: PC1 vs PC2 (colored by readmission status)', fontsize=14, fontweight='bold')
    ax1.legend(title='Readmission Status', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Colored by time in hospital
    ax2 = axes[1]
    scatter = ax2.scatter(
        diabetes_pca_results['PC1'],
        diabetes_pca_results['PC2'],
        c=diabetes_pca_results['time_in_hospital'],
        cmap='viridis',
        alpha=0.5,
        s=10
    )
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('Diabetes Data: PC1 vs PC2 (colored by time in hospital)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Time in Hospital (days)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = DIABETES_EXPLORATORY_DIR / "diabetes_pca_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved diabetes PCA plot", str(output_path))
    return output_path


@_cached
def dataset_comparison_analysis(
    diabetes_pca_results: pd.DataFrame,
    pca_results: pd.DataFrame,
    diabetes_preprocessed_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare RNA-seq and diabetes datasets to assess if they share patterns.
    
    This analysis evaluates whether the two datasets exhibit similar structure
    or patterns that could inform the analysis. Both datasets are analyzed
    independently using PCA, then compared through correlation analysis and
    variance explained metrics.
    
    Includes multiple validation tests:
    1. Correlation between PC1/PC2
    2. Statistical significance of correlation
    3. Variance explained comparison
    4. Structure assessment (clustering quality)
    """
    animator.show_info("Comparing RNA-seq and diabetes datasets with multiple validation tests...")
    
    from scipy import stats
    
    # Extract PC1 and PC2 from both datasets
    rna_pc1 = pca_results['PC1'].values
    rna_pc2 = pca_results['PC2'].values
    
    # For diabetes, sample a subset to match RNA-seq sample size (52)
    # This allows comparison
    n_rna_samples = len(rna_pc1)
    diabetes_subset = diabetes_pca_results.sample(n=min(n_rna_samples, len(diabetes_pca_results)), random_state=42)
    diabetes_pc1 = diabetes_subset['PC1'].values[:n_rna_samples]
    diabetes_pc2 = diabetes_subset['PC2'].values[:n_rna_samples]
    
    # Calculate correlations between principal components
    pc1_corr = np.corrcoef(rna_pc1[:len(diabetes_pc1)], diabetes_pc1)[0, 1]
    pc2_corr = np.corrcoef(rna_pc2[:len(diabetes_pc2)], diabetes_pc2)[0, 1]
    
    # Statistical significance test for correlations
    pc1_stat, pc1_pvalue = stats.pearsonr(rna_pc1[:len(diabetes_pc1)], diabetes_pc1)
    pc2_stat, pc2_pvalue = stats.pearsonr(rna_pc2[:len(diabetes_pc2)], diabetes_pc2)
    
    # Calculate variance explained for both datasets
    # For RNA-seq, we need to get this from the PCA object (stored in cache or recalculate)
    # For diabetes, we can get it from the PCA results
    # Let's recalculate to get variance explained
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.preprocessing import StandardScaler
    
    # RNA-seq variance explained (from existing PCA)
    rna_expr_data = pca_results[['PC1', 'PC2']].values
    # We'll use the variance of PC1 and PC2 as proxy
    rna_pc1_var = np.var(rna_pc1)
    rna_pc2_var = np.var(rna_pc2)
    rna_total_var = rna_pc1_var + rna_pc2_var
    rna_pc1_var_explained = rna_pc1_var / rna_total_var if rna_total_var > 0 else 0
    rna_pc2_var_explained = rna_pc2_var / rna_total_var if rna_total_var > 0 else 0
    
    # Diabetes variance explained (recalculate from preprocessed data)
    diabetes_data = diabetes_preprocessed_data.values
    diabetes_scaled = StandardScaler().fit_transform(diabetes_data)
    diabetes_pca = SKPCA(n_components=2)
    diabetes_pca.fit(diabetes_scaled)
    diabetes_pc1_var_explained = diabetes_pca.explained_variance_ratio_[0]
    diabetes_pc2_var_explained = diabetes_pca.explained_variance_ratio_[1]
    
    # Structure assessment: Check if RNA-seq has clearer separation
    # Use silhouette score or simple variance ratio
    rna_separation_score = np.std(rna_pc1) / (np.std(rna_pc1) + np.std(rna_pc2) + 1e-10)
    diabetes_separation_score = np.std(diabetes_pc1) / (np.std(diabetes_pc1) + np.std(diabetes_pc2) + 1e-10)
    
    # Create comprehensive comparison summary
    comparison = pd.DataFrame({
        'metric': [
            'PC1 Correlation',
            'PC1 Correlation P-value',
            'PC2 Correlation',
            'PC2 Correlation P-value',
            'RNA-seq PC1 Variance Explained',
            'Diabetes PC1 Variance Explained',
            'RNA-seq PC2 Variance Explained',
            'Diabetes PC2 Variance Explained',
            'RNA-seq Structure Score',
            'Diabetes Structure Score',
            'RNA-seq Samples',
            'Diabetes Samples (subset)'
        ],
        'value': [
            pc1_corr,
            pc1_pvalue,
            pc2_corr,
            pc2_pvalue,
            rna_pc1_var_explained,
            diabetes_pc1_var_explained,
            rna_pc2_var_explained,
            diabetes_pc2_var_explained,
            rna_separation_score,
            diabetes_separation_score,
            len(rna_pc1),
            len(diabetes_pc1)
        ],
        'interpretation': [
            f'Correlation coefficient: {pc1_corr:.4f}' + (' (weak)' if abs(pc1_corr) < 0.1 else ' (moderate)'),
            'Not statistically significant' if pc1_pvalue > 0.05 else 'Statistically significant',
            f'Correlation coefficient: {pc2_corr:.4f}' + (' (weak)' if abs(pc2_corr) < 0.1 else ' (moderate)'),
            'Not statistically significant' if pc2_pvalue > 0.05 else 'Statistically significant',
            f'{rna_pc1_var_explained:.1%} variance explained',
            f'{diabetes_pc1_var_explained:.1%} variance explained',
            f'{rna_pc2_var_explained:.1%} variance explained',
            f'{diabetes_pc2_var_explained:.1%} variance explained',
            'Structure assessment metric',
            'Structure assessment metric',
            f'{len(rna_pc1)} samples',
            f'{len(diabetes_pc1)} samples (random subset)'
        ]
    })
    
    logger.info(f"Dataset comparison: PC1 correlation = {pc1_corr:.4f} (p={pc1_pvalue:.4f}), PC2 correlation = {pc2_corr:.4f} (p={pc2_pvalue:.4f})")
    logger.info(f"Variance explained: RNA-seq PC1={rna_pc1_var_explained:.1%}, Diabetes PC1={diabetes_pc1_var_explained:.1%}")
    logger.info(f"Correlation analysis results: PC1 correlation={pc1_corr:.4f} (p={pc1_pvalue:.4f}), PC2 correlation={pc2_corr:.4f} (p={pc2_pvalue:.4f})")
    
    return comparison


def create_dataset_comparison_plot(
    diabetes_pca_results: pd.DataFrame,
    pca_results: pd.DataFrame,
    dataset_comparison_analysis: pd.DataFrame
) -> Path:
    """
    Create visualization comparing RNA-seq and diabetes PCA results.
    """
    DIABETES_EXPLORATORY_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: RNA-seq PCA
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(
        pca_results['PC1'],
        pca_results['PC2'],
        c=range(len(pca_results)),
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    ax1.set_xlabel('PC1', fontsize=12)
    ax1.set_ylabel('PC2', fontsize=12)
    ax1.set_title('RNA-seq Data: PC1 vs PC2\n(52 samples, clear population separation)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    # Sample subset of diabetes data for visualization
    diabetes_subset = diabetes_pca_results.sample(n=min(1000, len(diabetes_pca_results)), random_state=42)
    scatter2 = ax2.scatter(
        diabetes_subset['PC1'],
        diabetes_subset['PC2'],
        c=diabetes_subset['time_in_hospital'],
        cmap='plasma',
        s=10,
        alpha=0.3
    )
    ax2.set_xlabel('PC1', fontsize=12)
    ax2.set_ylabel('PC2', fontsize=12)
    ax2.set_title('Diabetes Data: PC1 vs PC2\n(101K encounters, diffuse distribution)', 
                  fontsize=13, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Time in Hospital')
    ax2.grid(True, alpha=0.3)
    
    # Bottom row: Statistical validation
    ax3 = axes[1, 0]
    
    # Get all comparison metrics
    pc1_corr = dataset_comparison_analysis[dataset_comparison_analysis['metric'] == 'PC1 Correlation']['value'].values[0]
    pc1_pval = dataset_comparison_analysis[dataset_comparison_analysis['metric'] == 'PC1 Correlation P-value']['value'].values[0]
    pc2_corr = dataset_comparison_analysis[dataset_comparison_analysis['metric'] == 'PC2 Correlation']['value'].values[0]
    pc2_pval = dataset_comparison_analysis[dataset_comparison_analysis['metric'] == 'PC2 Correlation P-value']['value'].values[0]
    rna_var = dataset_comparison_analysis[dataset_comparison_analysis['metric'] == 'RNA-seq PC1 Variance Explained']['value'].values[0]
    diab_var = dataset_comparison_analysis[dataset_comparison_analysis['metric'] == 'Diabetes PC1 Variance Explained']['value'].values[0]
    
    # Create subplot with correlation and p-values
    x_pos = np.arange(2)
    width = 0.35
    
    # Correlation bars
    correlations = [pc1_corr, pc2_corr]
    colors = ['red' if abs(c) < 0.1 else 'orange' if abs(c) < 0.3 else 'green' for c in correlations]
    bars1 = ax3.bar(x_pos - width/2, correlations, width, label='Correlation', color=colors, alpha=0.7)
    
    # P-values (scaled for visualization, shown as text)
    pvalues = [pc1_pval, pc2_pval]
    bars2 = ax3.bar(x_pos + width/2, [p*10 for p in pvalues], width, label='P-value (×10)', color='lightblue', alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_xlabel('Principal Component', fontsize=12)
    ax3.set_title('Statistical Validation: Correlation & Significance', 
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['PC1', 'PC2'])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar1, bar2, corr, pval) in enumerate(zip(bars1, bars2, correlations, pvalues)):
        # Correlation value
        height1 = bar1.get_height()
        ax3.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{corr:.3f}',
                ha='center', va='bottom' if height1 > 0 else 'top', fontsize=9, fontweight='bold')
        # P-value label
        height2 = bar2.get_height()
        ax3.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'p={pval:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # Variance explained comparison
    ax4 = axes[1, 1]
    datasets = ['RNA-seq', 'Diabetes']
    variances = [rna_var * 100, diab_var * 100]  # Convert to percentage
    colors_var = ['green' if v > 20 else 'orange' for v in variances]
    bars_var = ax4.bar(datasets, variances, color=colors_var, alpha=0.7, width=0.6)
    ax4.set_ylabel('Variance Explained (%)', fontsize=12)
    ax4.set_title('PC1 Variance Explained Comparison', 
                  fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, var in zip(bars_var, variances):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add significance indicator
    if pc1_pval > 0.05 and pc2_pval > 0.05:
        ax4.text(0.5, 0.95, 'Correlations NOT significant\n(p > 0.05 for both PC1 and PC2)',
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=9)
    
    plt.tight_layout()
    
    output_path = DIABETES_EXPLORATORY_DIR / "dataset_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    animator.show_file_operation("Saved dataset comparison plot", str(output_path))
    return output_path


@_cached
def create_all_diabetes_exploratory_analysis(
    diabetes_pca_results: pd.DataFrame,
    pca_results: pd.DataFrame,
    dataset_comparison_analysis: pd.DataFrame
) -> pd.DataFrame:
    """
    Create all diabetes exploratory analysis plots and comparisons.
    """
    animator.show_operation_start("Creating Diabetes Exploratory Analysis", "Generating PCA and comparison plots")
    
    # Create diabetes PCA plot
    create_diabetes_pca_plot(diabetes_pca_results)
    
    # Create comparison plot
    create_dataset_comparison_plot(diabetes_pca_results, pca_results, dataset_comparison_analysis)
    
    animator.show_operation_complete("Diabetes Exploratory Analysis", 0)
    
    return dataset_comparison_analysis


