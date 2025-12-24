# Hamilton Interview Exercise

Data science interview exercise using Hamilton DAG framework with real clinical and biological datasets.

## Overview

This project is adapted from the [Phase-Zero-Labs interview template](https://github.com/Phase-Zero-Labs/pzl-interview-template) to work within the Analyses workspace, using shared utilities and following workspace conventions.

**Datasets**:
- Diabetes 130-US Hospitals (1999-2008) - 101K patient encounters
- RNA-seq drug screening - mABs and other compounds exposed to skin organoids

**Time**: ~1 hour  
**Tools**: This repo + AI assistant (Claude Code or similar)

## 📊 Final Analysis Report

**Main Deliverables:**
- **[FINAL_REPORT.md](FINAL_REPORT.md)** - Complete analysis report (markdown)
- **[FINAL_REPORT.pdf](FINAL_REPORT.pdf)** - Professional PDF report with all figures embedded
- **[docs/compliance/FINAL_COMPLIANCE_CHECK.md](docs/compliance/FINAL_COMPLIANCE_CHECK.md)** - Compliance verification

The report includes:
- Executive summary of findings
- Complete analysis journey (AI collaboration process)
- Key findings with visualizations
- Biological interpretation
- Technical details and Hamilton DAG structure

## What This Exercise Evaluates

This is **not** a test of what you know. The focus is on:

1. **How you collaborate with AI** - Your prompting strategy, iteration process, and knowing when to trust (or question) AI output
2. **Your analytical thinking** - How you explore data, form hypotheses, and validate findings
3. **Communication** - Talking through your reasoning as you work

There are no trick questions. The datasets are messy. Perfect answers don't exist.

## Prerequisites

- **Python 3.11+**
- **Shared workspace venv** - Uses the workspace's shared virtual environment
- **Hamilton** - Will be installed via requirements.txt

## Quick Start

```bash
# Navigate to project
cd projects/hamilton-interview

# Activate shared workspace venv (if not already activated)
# From workspace root:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "from modules.diabetes import raw_diabetic_data; print(raw_diabetic_data().shape)"
# Should print: (101766, 50)

# List available nodes
python main.py --list

# Run a specific node
python main.py --outputs raw_diabetic_data

# Run analysis with dependencies
python main.py --outputs readmission_by_age

# Visualize DAG
python main.py --visualize
```

## Project Structure

```
hamilton-interview/
├── main.py                 # Main entry point
├── modules/
│   ├── __init__.py
│   ├── hamilton_driver.py  # Hamilton driver wrapper
│   ├── diabetes.py         # Diabetes dataset pipeline
│   └── rna.py              # RNA-seq dataset pipeline
├── input/                  # Input data files
│   ├── diabetic_data.csv
│   ├── IDS_mapping.csv
│   ├── salmon_gene_counts.tsv
│   └── Samples ID.xlsx
├── output/                 # Results and cache
│   └── cache/             # Cached parquet files
├── requirements.txt        # Project dependencies
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

## The Datasets

### Diabetes (modules/diabetes.py)

Clinical data from 130 US hospitals. Each row is a diabetic patient encounter.

| Attribute | Value |
|-----------|-------|
| Rows | 101,766 patient encounters |
| Columns | 50 features |
| Target | `readmitted` (<30 days / >30 days / No) |

Key features:
- **Demographics**: race, gender, age (binned by decade)
- **Encounter**: admission_type, discharge_disposition, time_in_hospital
- **Clinical**: num_lab_procedures, num_medications, number_diagnoses
- **Medications**: 23 drug columns showing dosage changes (Up/Down/Steady/No)

### RNA-seq (modules/rna.py)

Gene expression from drug screening experiments - mABs and other compounds exposed at multiple concentrations to skin organoids.

| Attribute | Value |
|-----------|-------|
| Samples | 52 across 3 plates |
| Genes | 78,932 |
| Compounds | 14 (ADCs, free drugs, controls) |

## Data Quality Notes

This is real-world data with real-world messiness:
- Missing values encoded as `?` (not NaN)
- High missingness in `weight`, `medical_specialty`, `payer_code`
- Class imbalance in readmission target (~54% No, ~35% >30, ~11% <30)
- Diagnosis codes are raw ICD-9

## Hamilton DAG Pattern

The repo uses Hamilton for DAG-based pipelines. Functions with matching parameter names create edges:

```python
def raw_diabetic_data() -> pd.DataFrame:
    """Load the dataset."""
    return pd.read_csv("input/diabetic_data.csv")

def readmission_by_age(raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """Parameter 'raw_diabetic_data' matches function above.
    Edge created: raw_diabetic_data -> readmission_by_age"""
    return raw_diabetic_data.groupby('age')['readmitted'].value_counts()
```

## Available Nodes

### Diabetes

| Node | Description |
|------|-------------|
| `raw_diabetic_data` | Load raw CSV (101K rows, 50 cols) |
| `admission_type_lookup` | ID to description mapping (8 types) |
| `discharge_disposition_lookup` | ID to description mapping (30 types) |
| `admission_source_lookup` | ID to description mapping (25 types) |
| `readmission_by_age` | Example analysis: readmission rates by age |

### RNA-seq

| Node | Description |
|------|-------------|
| `sample_metadata` | Sample info (52 samples, plate/compound/concentration) |
| `raw_gene_counts` | Salmon gene counts (78K genes x 52 samples) |

## CLI Reference

```bash
python main.py --list                        # See available nodes
python main.py --outputs raw_diabetic_data   # Run specific node
python main.py --outputs readmission_by_age  # Runs dependencies too
python main.py --visualize                   # Show DAG
```

## Caching

Pipeline functions use the `@_cached` decorator which:
- Saves DataFrame outputs as parquet files to `output/cache/`
- Enables faster re-runs by loading from cache
- Files are named `{function_name}.parquet`

To clear the cache:
```bash
rm -rf output/cache/*.parquet  # Unix/Linux/macOS
Remove-Item output/cache/*.parquet  # Windows PowerShell
```

## Workspace Integration

This project uses workspace utilities:
- **File operations**: `utils.file_utils` for loading/saving data
- **Animations**: `utils.animation_utils` for user-facing operations
- **Debugging**: `utils.debug_utils` for logging and system information
- **Shared venv**: Uses the workspace's shared virtual environment

## Getting Started with Analysis

1. **Explore the data**:
   ```python
   from modules.diabetes import raw_diabetic_data
   df = raw_diabetic_data()
   print(df.info())
   print(df.describe())
   ```

2. **Create new analysis nodes**:
   - Add functions to `modules/diabetes.py` or `modules/rna.py`
   - Use parameter names that match existing function names to create DAG edges
   - Use `@_cached` decorator for DataFrame outputs

3. **Run your analysis**:
   ```bash
   python main.py --outputs your_new_function
   ```

## Example: Creating a New Analysis Node

```python
@_cached
def readmission_by_gender(raw_diabetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Readmission rates broken down by gender.
    
    The parameter name 'raw_diabetic_data' matches the function above,
    so Hamilton creates the edge: raw_diabetic_data -> readmission_by_gender
    """
    summary = raw_diabetic_data.groupby('gender')['readmitted'].value_counts(normalize=True)
    summary = summary.unstack(fill_value=0) * 100
    summary = summary.round(2).reset_index()
    summary.columns.name = None
    return summary
```

Then run:
```bash
python main.py --outputs readmission_by_gender
```

## Notes

- This project is adapted to work within the Analyses workspace
- Uses workspace utilities for consistency with other projects
- Follows workspace conventions (input/, output/, modules/ structure)
- Hamilton handles dependency resolution automatically

## License

Adapted from Phase-Zero-Labs interview template. Original template: https://github.com/Phase-Zero-Labs/pzl-interview-template


