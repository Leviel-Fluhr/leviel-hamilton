"""
Hamilton Driver Wrapper

Adapted to use workspace utilities and conventions.
"""

import importlib
from pathlib import Path
from typing import Optional

from hamilton import driver

# Workspace import setup
import sys
workspace_root = Path(__file__).parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
from workspace.imports import setup_workspace_imports
setup_workspace_imports(__file__)

from utils.animation_utils import animator
from utils.debug_utils import quick_debug_setup

logger, config = quick_debug_setup(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def discover_modules():
    """Discover Hamilton modules in modules/ directory."""
    modules_dir = PROJECT_ROOT / "modules"
    modules = []

    # Look for .py files in modules/ (not in subdirectories)
    for py_file in modules_dir.glob("*.py"):
        if py_file.name in ("__init__.py", "hamilton_driver.py"):
            continue

        module_name = f"modules.{py_file.stem}"
        try:
            module = importlib.import_module(module_name)
            modules.append(module)
            logger.info(f"Loaded module: {module_name}")
        except ImportError as e:
            animator.show_warning(f"Could not import {module_name}: {e}")

    return modules


def build_driver(track: bool = False) -> Optional[driver.Driver]:
    """Build Hamilton driver with discovered modules."""
    modules = discover_modules()

    if not modules:
        animator.show_error(
            "Hamilton Driver",
            "No Hamilton modules found in modules/"
        )
        animator.show_info("Add .py files with functions to modules/ to build your pipeline")
        return None

    builder = driver.Builder().with_modules(*modules)

    if track:
        try:
            from hamilton.plugins import h_tracker

            tracker = h_tracker.HamiltonTracker(
                project_id="hamilton-interview",
                username="user",
                dag_name="pipeline",
            )
            builder = builder.with_adapters(tracker)
            animator.show_info("Hamilton UI tracking enabled")
        except ImportError:
            animator.show_warning(
                "Hamilton tracking not available. Install with: pip install sf-hamilton[ui,sdk]"
            )

    return builder.build()


def list_outputs(dr: driver.Driver):
    """List all available outputs from the pipeline."""
    modules = discover_modules()

    if not modules:
        animator.show_error(
            "Hamilton Driver",
            "No Hamilton modules found in modules/"
        )
        return

    animator.show_info("Available outputs:")
    print("-" * 40)

    for module in modules:
        module_name = module.__name__.split(".")[-1]
        funcs = [
            name
            for name in dir(module)
            if not name.startswith("_") and callable(getattr(module, name))
        ]
        if funcs:
            print(f"\n{module_name}:")
            for func in funcs:
                print(f"  - {func}")


def run_pipeline(dr: driver.Driver, outputs: list[str]) -> int:
    """Run pipeline with specified outputs."""
    from utils.file_utils import ensure_output_dir

    # Ensure output directories exist
    ensure_output_dir()
    (PROJECT_ROOT / "output" / "cache").mkdir(parents=True, exist_ok=True)

    animator.show_operation_start(
        "Running Hamilton pipeline",
        f"Computing outputs: {', '.join(outputs)}"
    )

    try:
        results = dr.execute(final_vars=outputs)

        animator.show_operation_complete("Hamilton pipeline", 0)

        print("\n" + "=" * 40)
        print("PIPELINE COMPLETE")
        print("=" * 40)
        print(f"Computed outputs: {list(results.keys())}")

        # Show summary for DataFrames
        for name, result in results.items():
            if hasattr(result, "shape"):
                print(f"  {name}: {result.shape[0]:,} rows x {result.shape[1]} cols")
            elif isinstance(result, dict):
                print(f"  {name}: {len(result)} items")

        return 0

    except Exception as e:
        animator.show_error("Pipeline Execution", str(e))
        import traceback
        traceback.print_exc()
        return 1


