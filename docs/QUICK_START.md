# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Make sure you're in the project directory
cd projects/hamilton-interview

# Activate shared workspace venv (if not already active)
# From workspace root:
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/Linux/macOS

# Install project dependencies
pip install -r requirements.txt
```

### Step 2: Verify Setup

```bash
# Run verification script
python verify_setup.py
```

This will check:
- ✅ Data files are present
- ✅ Dependencies are installed
- ✅ Hamilton modules can be imported
- ✅ Hamilton driver can be built

### Step 3: Run Your First Node

```bash
# List all available nodes
python main.py --list

# Run a simple data loading node
python main.py --outputs raw_diabetic_data

# Run an analysis node (will run dependencies automatically)
python main.py --outputs readmission_by_age
```

## 📊 Available Commands

```bash
# List all available pipeline nodes
python main.py --list

# Run specific node(s)
python main.py --outputs raw_diabetic_data
python main.py --outputs readmission_by_age sample_metadata

# Visualize the DAG (requires: pip install sf-hamilton[visualization])
python main.py --visualize

# Get help
python main.py --help
```

## 🔍 Exploring the Data

### Python Interactive Session

```python
# Start Python in the project directory
python

# Import and explore
from modules.diabetes import raw_diabetic_data
from modules.rna import sample_metadata, raw_gene_counts

# Load data
df_diabetes = raw_diabetic_data()
print(df_diabetes.shape)
print(df_diabetes.info())
print(df_diabetes.head())

# Explore RNA-seq data
metadata = sample_metadata()
print(metadata.head())
print(metadata['compound'].value_counts())
```

## 🎯 Creating Your First Analysis Node

### Example: Readmission by Gender

Add this to `modules/diabetes.py`:

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

## 📁 Project Structure

```
hamilton-interview/
├── main.py                 # Entry point
├── verify_setup.py         # Setup verification script
├── modules/
│   ├── diabetes.py         # Diabetes dataset pipeline
│   ├── rna.py              # RNA-seq dataset pipeline
│   └── hamilton_driver.py  # Hamilton driver wrapper
├── input/                  # Data files (4 files)
├── output/
│   └── cache/             # Cached parquet files
└── requirements.txt        # Dependencies
```

## 🐛 Troubleshooting

### "Module not found" errors
- Make sure workspace venv is activated
- Run `pip install -r requirements.txt`

### "File not found" errors
- Check that data files are in `input/` directory
- Run `python verify_setup.py` to check

### Hamilton import errors
- Install: `pip install sf-hamilton>=1.52.0`
- Check: `python -c "import hamilton; print(hamilton.__version__)"`

### Visualization not working
- Install: `pip install sf-hamilton[visualization]`
- Or skip visualization and use `--list` and `--outputs` commands

## 📚 Next Steps

1. **Explore the data** - Use Python to understand the datasets
2. **Create analysis nodes** - Build your own Hamilton functions
3. **Connect nodes** - Use parameter names to create DAG edges
4. **Run analyses** - Execute your pipeline with `python main.py --outputs <node>`

## 💡 Tips

- **Hamilton DAG**: Functions become nodes, parameter names create edges
- **Caching**: All `@_cached` functions save to `output/cache/` as parquet
- **Workspace utilities**: Use `utils.file_utils`, `utils.animation_utils`, etc.
- **Type hints**: Hamilton uses type hints to understand data flow

## 🎓 Learning Resources

- Hamilton docs: https://hamilton.dagworks.io/
- Original template: https://github.com/Phase-Zero-Labs/pzl-interview-template
- Workspace README: `../../README.md`

