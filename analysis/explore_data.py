#!/usr/bin/env python
"""
Quick data exploration script for the interview exercise.

Run this to understand the datasets before building analysis nodes.
"""

import sys
from pathlib import Path

# Workspace import setup
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.animation_utils import animator
from modules.diabetes import raw_diabetic_data
from modules.rna import sample_metadata, raw_gene_counts

animator.show_operation_start("Exploring Datasets", "Loading and examining data")

print("\n" + "="*60)
print("DIABETES DATASET")
print("="*60)

df_diabetes = raw_diabetic_data()
print(f"\nShape: {df_diabetes.shape[0]:,} rows x {df_diabetes.shape[1]} columns")
print(f"\nColumns: {list(df_diabetes.columns)}")
print(f"\nFirst few rows:")
print(df_diabetes.head())
print(f"\nData types:")
print(df_diabetes.dtypes)
print(f"\nMissing values (encoded as '?'):")
missing = (df_diabetes == '?').sum()
print(missing[missing > 0])
print(f"\nTarget variable 'readmitted' distribution:")
print(df_diabetes['readmitted'].value_counts())
print(f"\nAge distribution:")
print(df_diabetes['age'].value_counts().sort_index())
print(f"\nGender distribution:")
print(df_diabetes['gender'].value_counts())

print("\n" + "="*60)
print("RNA-SEQ DATASET - SAMPLE METADATA")
print("="*60)

metadata = sample_metadata()
print(f"\nShape: {metadata.shape[0]} samples x {metadata.shape[1]} columns")
print(f"\nColumns: {list(metadata.columns)}")
print(f"\nFirst few rows:")
print(metadata.head(10))
print(f"\nCompounds tested:")
print(metadata['compound'].value_counts())
print(f"\nPlates:")
print(metadata['plate'].value_counts())
print(f"\nConcentration ranges:")
print(metadata.groupby('compound')['concentration'].agg(['min', 'max', 'mean']))

print("\n" + "="*60)
print("RNA-SEQ DATASET - GENE COUNTS")
print("="*60)

gene_counts = raw_gene_counts()
print(f"\nShape: {gene_counts.shape[0]:,} genes x {gene_counts.shape[1]} columns")
print(f"\nFirst few columns: {list(gene_counts.columns[:5])}...")
print(f"\nFirst few rows:")
print(gene_counts.head())
print(f"\nSample columns (TD009_*): {len([c for c in gene_counts.columns if c.startswith('TD009_')])} samples")

animator.show_operation_complete("Data Exploration", 0)
print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Think about interesting questions to answer")
print("2. Create new analysis nodes in modules/diabetes.py or modules/rna.py")
print("3. Use parameter names to connect nodes in the DAG")
print("4. Run your analysis: python main.py --outputs your_function_name")
print("="*60)

