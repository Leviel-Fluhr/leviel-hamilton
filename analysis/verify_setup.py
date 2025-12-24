#!/usr/bin/env python
"""
Quick verification script to test project setup.

Run this after installing dependencies to verify everything works.
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
from utils.debug_utils import quick_debug_setup

logger, config = quick_debug_setup(__name__)

PROJECT_ROOT = Path(__file__).parent


def check_data_files():
    """Check that all required data files exist."""
    animator.show_info("Checking data files...")
    
    required_files = [
        "input/diabetic_data.csv",
        "input/IDS_mapping.csv",
        "input/salmon_gene_counts.tsv",
        "input/Samples ID.xlsx",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size = full_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  [OK] {file_path} ({size:.2f} MB)")
        else:
            print(f"  [FAIL] {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check that required packages are installed."""
    animator.show_info("Checking dependencies...")
    
    required_packages = [
        "hamilton",
        "pandas",
        "pyarrow",
        "openpyxl",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [FAIL] {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n  Install missing packages: pip install {' '.join(missing)}")
        return False
    
    return True


def check_hamilton_modules():
    """Check that Hamilton modules can be imported."""
    animator.show_info("Checking Hamilton modules...")
    
    try:
        from modules import diabetes, rna
        print("  [OK] modules.diabetes")
        print("  [OK] modules.rna")
        
        # Check for key functions
        if hasattr(diabetes, 'raw_diabetic_data'):
            print("  [OK] diabetes.raw_diabetic_data function exists")
        if hasattr(rna, 'sample_metadata'):
            print("  [OK] rna.sample_metadata function exists")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Could not import modules: {e}")
        return False


def check_hamilton_driver():
    """Check that Hamilton driver can be built."""
    animator.show_info("Checking Hamilton driver...")
    
    try:
        from modules.hamilton_driver import build_driver
        
        dr = build_driver()
        if dr is None:
            print("  [FAIL] Driver is None")
            return False
        
        print("  [OK] Hamilton driver built successfully")
        
        # Try to get available nodes
        nodes = dr.list_available_variables()
        print(f"  [OK] Found {len(nodes)} available nodes")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Could not build driver: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    animator.show_operation_start("Verifying Project Setup", "Running verification checks")
    
    checks = [
        ("Data Files", check_data_files),
        ("Dependencies", check_dependencies),
        ("Hamilton Modules", check_hamilton_modules),
        ("Hamilton Driver", check_hamilton_driver),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  [ERROR] {name} check failed: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        animator.show_operation_complete("Project Setup Verification", 0)
        animator.show_info("All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("  python main.py --list                        # See available nodes")
        print("  python main.py --outputs raw_diabetic_data   # Run a node")
        return 0
    else:
        animator.show_warning("Some checks failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    exit(main() or 0)

