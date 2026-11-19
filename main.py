"""Thin entry point: forwards to the Phase 3 pipeline in run_phase3.py.

    python main.py --data-root /path/to/datasets --dry-run
"""

from run_phase3 import main

if __name__ == "__main__":
    main()
