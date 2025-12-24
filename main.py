#!/usr/bin/env python
"""
Hamilton Interview Exercise - Main Entry Point

Adapted from pzl-interview-template to use workspace utilities.
"""

import sys
import argparse
from pathlib import Path

# Workspace import setup
workspace_root = Path(__file__).parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.animation_utils import animator
from utils.debug_utils import quick_debug_setup
from modules.hamilton_driver import build_driver, list_outputs, run_pipeline

logger, config = quick_debug_setup(__name__)


def main():
    """Main entry point for Hamilton pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Hamilton pipeline for interview exercise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list                        # See available nodes
  python main.py --outputs raw_diabetic_data   # Run specific node
  python main.py --outputs readmission_by_age  # Runs dependencies too
  python main.py --visualize                   # Show DAG
        """
    )
    parser.add_argument(
        "--outputs", nargs="+", help="Specific outputs to compute"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available outputs"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show DAG visualization"
    )

    args = parser.parse_args()

    # Build driver
    dr = build_driver()
    if dr is None:
        return 1

    # Handle list mode
    if args.list:
        list_outputs(dr)
        return 0

    # Handle visualize mode
    if args.visualize:
        animator.show_info("Displaying DAG visualization...")
        try:
            dr.display_all_functions()
        except Exception as e:
            animator.show_error("Visualization", str(e))
            animator.show_info("Try: pip install sf-hamilton[visualization]")
        return 0

    # Handle execution mode
    if args.outputs:
        return run_pipeline(dr, args.outputs)
    else:
        parser.print_help()
        animator.show_info("Tip: Use --list to see available outputs")
        return 0


if __name__ == "__main__":
    exit(main() or 0)


