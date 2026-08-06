# Hamilton  : RNA-seq Drug Screening Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Hamilton](https://img.shields.io/badge/Hamilton-DAG%20Framework-green.svg)](https://github.com/dagworks-inc/hamilton)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Complete analysis of RNA-seq drug screening data identifying two distinct populations based on epithelial activation pathways**

## 📊 Overview

This project presents a comprehensive analysis of RNA-seq gene expression data from drug screening experiments on skin organoids. Using the **Hamilton DAG framework**, the analysis pipeline identifies and validates two distinct biological populations driven by epithelial activation and keratinization pathways.

### Key Findings

- **Two distinct populations** identified through PCA analysis
- **970 genes** significantly different between populations (p < 0.05)
- **20 significant pathways** enriched (FDR < 0.05), primarily related to keratinization
- **Biological interpretation**: ADCs and free cytotoxic drugs induce stress response in skin organoids

## 🎯 Main Deliverables

- **[FINAL_REPORT.pdf](FINAL_REPORT.pdf)** - Professional PDF report with all visualizations
- **[FINAL_REPORT.md](FINAL_REPORT.md)** - Complete analysis report (markdown)
- **11 publication-quality plots** - All analysis visualizations (including evidence-based dataset comparison)
- **Complete Hamilton DAG pipeline** - Reproducible analysis workflow

## 📈 Analysis Highlights

### 1. Exploratory Analysis
- Principal Component Analysis revealing two distinct clusters
- Sample correlation matrix showing population structure
- Compound expression comparison across treatments
- Diabetes dataset exploration and comparison with RNA-seq

### 2. Population Analysis
- Differential expression analysis (970 significant genes)
- Statistical validation (t-tests, Mann-Whitney U tests)
- Feature importance ranking (top separating genes)

### 3. Pathway Enrichment
- GO, KEGG, and Reactome enrichment analysis
- 20 significant pathways identified
- Biological validation through literature search

### 4. Visualization
- 11 publication-quality figures (including diabetes comparison)
- Analysis pathway summary diagram
- Professional PDF report

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Shared workspace venv** (or create your own virtual environment)
- **Hamilton** (installed via requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/Leviel-Fluhr/leviel-hamilton.git
cd leviel-hamilton

# Activate virtual environment (if using shared workspace)
# From workspace root:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# List available Hamilton nodes
python main.py --list

# Run exploratory analysis
python main.py --outputs create_all_exploratory_plots

# Run population analysis
python main.py --outputs create_all_population_analysis_plots

# Run pathway enrichment
python main.py --outputs create_all_pathway_analysis

# Visualize DAG structure
python main.py --visualize
```

## 📁 Project Structure

```
hamilton-interview/
├── README.md                    # This file
├── FINAL_REPORT.md              # Complete analysis report
├── FINAL_REPORT.pdf             # PDF report with all figures
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
│
├── modules/                     # Hamilton DAG modules
│   ├── diabetes.py             # Diabetes dataset pipeline
│   ├── rna.py                  # RNA-seq analysis pipeline (main work)
│   └── hamilton_driver.py      # Hamilton driver wrapper
│
├── analysis/                    # Analysis scripts
│   ├── explore_data.py         # Data exploration utility
│   ├── verify_setup.py         # Setup verification
│   └── scripts/                # Utility scripts
│       ├── create_summary_viz.py
│       └── generate_pdf_report.py
│
├── input/                       # Input data files
│   ├── diabetic_data.csv
│   ├── salmon_gene_counts.tsv
│   ├── Samples ID.xlsx
│   └── IDS_mapping.csv
│
├── output/                      # Analysis results
│   ├── exploratory/           # RNA-seq exploratory plots
│   ├── diabetes_exploratory/  # Diabetes dataset comparison plots
│   ├── population_analysis/   # Population analysis results
│   ├── pathway_enrichment/    # Pathway enrichment results
│   ├── visualizations/        # Summary visualizations
│   └── cache/                 # Cached intermediate results
│
└── docs/                       # Documentation
    ├── compliance/            # Compliance documentation
    └── [historical docs]      # Process documentation
```

## 🔬 Datasets

### RNA-seq Drug Screening
- **52 samples** across 3 plates
- **78,932 genes** quantified via Salmon
- **14 compounds** tested (ADCs, free drugs, controls)
- **Multiple concentrations** per compound

### Diabetes 130-US Hospitals
- **101,766 patient encounters**
- **50 features** (demographics, clinical, medications)
- **Target**: Readmission prediction

## 🏗️ Hamilton DAG Framework

This project uses **Hamilton** for building dataflows where functions define nodes and parameter names define edges:

```python
def raw_gene_counts() -> pd.DataFrame:
    """Load gene counts."""
    return pd.read_csv("input/salmon_gene_counts.tsv")

def log_transformed_expression(raw_gene_counts: pd.DataFrame) -> pd.DataFrame:
    """Log2 transform expression.
    
    Parameter 'raw_gene_counts' matches function above.
    Hamilton creates edge: raw_gene_counts -> log_transformed_expression
    """
    return np.log2(raw_gene_counts + 1)
```

**Key Features:**
- **Automatic dependency resolution** - Hamilton handles the DAG
- **Caching** - `@_cached` decorator caches expensive operations
- **Modularity** - Each function does one thing
- **Reproducibility** - Clear dependency chain

## 📊 Results Summary

### Dataset Evaluation
- **RNA-seq**: Clear structure (70.7% PC1 variance) - selected for detailed analysis
- **Diabetes**: Diffuse structure (14.1% PC1 variance) - no clear patterns for population analysis
- **Evidence-based decision**: Statistical comparison justified focusing on RNA-seq

### Population Separation
- Clear separation along PC1 (25.0% variance explained, 70.7% total PC1 variance)
- Two distinct groups: Controls/Dexamethasone vs. ADCs/Free drugs

### Top Separating Genes
All top genes are **epithelial structural proteins**:
- Keratins: KRT1, KRT5, KRT6A, KRT13, KRT14, KRT16, KRT17, KRT19
- Desmosomes: DSG1, DSG3, PKP1, DSC2, DSC3
- Cornified envelope: SPRR1A, SPRR1B, SPRR2A

### Pathway Enrichment
**Top 5 Significant Pathways:**
1. Keratinization (Reactome) - FDR = 6.48e-26
2. Cornified Envelope Formation (GO) - FDR = 1.28e-16
3. Intermediate Filament Organization (GO) - FDR = 1.85e-14
4. Formation of Cornified Envelope (Reactome) - FDR = 4.33e-14
5. Keratin Filament (GO) - FDR = 6.00e-13

### Biological Interpretation
- **Population 1**: Baseline epithelial state (controls/Dexamethasone)
- **Population 2**: Activated stress response state (ADCs/free drugs)
- **Mechanism**: Cytotoxic compounds induce keratinization and epithelial barrier remodeling

## 🛠️ Technical Details

### Dependencies
- `sf-hamilton>=1.52.0` - Hamilton DAG framework
- `pandas>=2.0.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning (PCA, statistical tests)
- `gseapy>=1.0.0` - Pathway enrichment analysis
- `matplotlib>=3.8.0`, `seaborn>=0.12.0` - Visualization
- `reportlab>=4.0.0` - PDF generation

See [requirements.txt](requirements.txt) for complete list.

### Workspace Integration
This project integrates with the Analyses workspace:
- Uses shared workspace utilities (`utils.file_utils`, `utils.animation_utils`, `utils.debug_utils`)
- Follows workspace conventions (input/, output/, modules/ structure)
- Uses shared virtual environment

## 📚 Documentation

- **[FINAL_REPORT.pdf](FINAL_REPORT.pdf)** - Complete analysis report with all figures
- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/compliance/](docs/compliance/)** - Compliance verification
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Detailed project structure

## 🎓 Interview Exercise Context

This project was completed as part of the **Hamilton Interview Exercise**, demonstrating:

1. **AI Collaboration** - Iterative process with clear reasoning documented
2. **Analytical Thinking** - Hypothesis-driven approach with multiple validations
3. **Communication** - Professional documentation and clear interpretation

**Evaluation Criteria:**
- How you collaborate with AI
- Your analytical thinking process
- Communication of reasoning

## 📄 License

Adapted from [Phase-Zero-Labs interview template](https://github.com/Phase-Zero-Labs/pzl-interview-template).

## 👤 Author

**Leviel Fluhr**

## 🔗 Links

- **Repository**: https://github.com/Leviel-Fluhr/leviel-hamilton
- **Original Template**: https://github.com/Phase-Zero-Labs/pzl-interview-template
- **Hamilton Framework**: https://github.com/dagworks-inc/hamilton

---

**Status**: ✅ Complete and ready for evaluation

**Last Updated**: December 2025
