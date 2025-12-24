# Project Structure

## 📁 Organized Folder Layout (Workspace Guidelines Compliant)

```
hamilton-interview/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── main.py                     # Main entry point
│
├── FINAL_REPORT.md             # ⭐ Main analysis report (markdown)
├── FINAL_REPORT.pdf            # ⭐ Main analysis report (PDF with figures)
│
├── docs/                       # All documentation files
│   ├── README.md              # Documentation index
│   ├── PROJECT_SUMMARY.md      # Quick project overview
│   ├── PROJECT_STRUCTURE.md    # This file
│   ├── compliance/            # Compliance documentation
│   │   ├── FINAL_COMPLIANCE_CHECK.md
│   │   └── SUBMISSION_CHECKLIST.md
│   ├── GUIDELINE_COMPLIANCE_AUDIT.md
│   ├── FINAL_RECOMMENDATION.md
│   └── [historical analysis docs]  # Process documentation
│
├── analysis/                   # Analysis and utility scripts
│   ├── explore_data.py        # Initial data exploration
│   ├── verify_setup.py        # Setup verification
│   └── scripts/               # Utility scripts
│       ├── create_summary_viz.py    # Generate summary visualization
│       └── generate_pdf_report.py   # Generate PDF report
│
├── modules/                    # Hamilton DAG modules
│   ├── __init__.py
│   ├── hamilton_driver.py     # Hamilton driver wrapper
│   ├── diabetes.py            # Diabetes dataset nodes
│   └── rna.py                 # RNA-seq analysis nodes (main work)
│
├── input/                      # Input data files
│   ├── diabetic_data.csv
│   ├── salmon_gene_counts.tsv
│   ├── Samples ID.xlsx
│   └── IDS_mapping.csv
│
└── output/                     # All analysis outputs
    ├── cache/                 # Cached Hamilton node results (parquet)
    ├── exploratory/           # Initial exploration plots
    │   ├── pca_plot.png
    │   ├── correlation_heatmap.png
    │   └── compound_comparison_heatmap.png
    ├── population_analysis/   # Population separation analysis
    │   ├── differential_expression/
    │   │   └── population_separation_top_genes.png
    │   ├── feature_importance/
    │   │   └── feature_importance_top_genes.png
    │   └── statistical_tests/
    │       └── volcano_plot.png
    ├── pathway_enrichment/    # Pathway enrichment results
    │   ├── pathway_enrichment_top_pathways.png
    │   ├── pathway_enrichment_results.csv
    │   └── significant_pathways.csv
    ├── visualizations/        # Summary visualizations
    │   └── population_summary.png
    └── analysis_pathway_summary.png  # Analysis pathway summary
```

## 🎯 Organization Principles (Workspace Guidelines)

1. **Root Level** → Only essential files:
   - `README.md` - Project overview
   - `main.py` - Main entry point
   - `requirements.txt` - Dependencies
   - `.gitignore` - Git ignore rules
   - `FINAL_REPORT.md` and `FINAL_REPORT.pdf` - Main deliverables

2. **Documentation** → `docs/` folder:
   - All documentation files organized by type
   - Compliance docs in `docs/compliance/`
   - Historical process docs preserved

3. **Analysis Scripts** → `analysis/` folder:
   - Main analysis scripts in `analysis/`
   - Utility scripts in `analysis/scripts/`

4. **Hamilton Modules** → `modules/` folder:
   - All DAG nodes organized by dataset/functionality

5. **Outputs** → Organized by analysis type in `output/`:
   - Logical subdirectories for each analysis step
   - Cache files in `output/cache/`

6. **No Project-Specific Venv** → Uses shared workspace venv:
   - No `.venv/` directory in project
   - Uses `../../venv/` (shared workspace venv)

## 📋 Key Files

### Main Deliverables
- `FINAL_REPORT.md` - Complete analysis report
- `FINAL_REPORT.pdf` - PDF version with all figures

### Entry Points
- `main.py` - Run Hamilton pipeline: `python main.py --outputs <node_name>`

### Utility Scripts
- `analysis/scripts/create_summary_viz.py` - Generate analysis pathway summary
- `analysis/scripts/generate_pdf_report.py` - Generate PDF report

### Core Analysis
- `modules/rna.py` - Main RNA-seq analysis pipeline (Hamilton DAG)
- `modules/diabetes.py` - Diabetes dataset pipeline
- `modules/hamilton_driver.py` - Hamilton driver wrapper

## ✅ Workspace Guidelines Compliance

- ✅ **No .venv/** - Uses shared workspace venv
- ✅ **Organized docs/** - All documentation in docs/ folder
- ✅ **Analysis scripts** - In analysis/ folder
- ✅ **Clean root** - Only essential files at root
- ✅ **Organized output/** - Logical subdirectories
- ✅ **Modules/** - Hamilton DAG modules properly organized
