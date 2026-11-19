# scripts/run_benchmark.py
"""
Compatibility wrapper around the real Phase 3 pipeline (run_phase3.py).

This script used to be an empty template that built a Phase3Runner with no
models, explainers or datasets and therefore did nothing. It now forwards all
arguments to run_phase3.main, so these are equivalent:

    python scripts/run_benchmark.py --data-root /path/to/datasets --dry-run
    python run_phase3.py           --data-root /path/to/datasets --dry-run
    vit-bench                      --data-root /path/to/datasets --dry-run
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_phase3 import main  # noqa: E402

if __name__ == "__main__":
    main()
