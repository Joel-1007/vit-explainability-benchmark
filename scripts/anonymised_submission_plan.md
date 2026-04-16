# Anonymised Submission Plan (TPAMI)

This document outlines the workflow to prepare the `vit-explainability-benchmark` repository for a double-blind peer review submission (Task 5.4.1).

## 1. Preparing the Codebase
- Disable all personal identifiers (author names in file headers, contact information, personal website links) across `pyproject.toml`, `.py` scripts, and Markdown files.
- The default setup specifies `Authors: Anonymous Authors` in `pyproject.toml`. Keep this intact.
- Avoid mentioning the institution (e.g. paths containing `C:\Users\Jayan\College\Research\` should not appear in final uploaded artifacts or code docs). A universal default like `/data/` is strictly utilized in scripts (e.g., `run_benchmark.sh`).

## 2. Setting Up the Anonymised Repository
- Create a new, throwaway GitHub/GitLab account specifically for the submission (e.g., `vit-bench-anon`).
- Push a **clean** branch of the repository. Do **not** push the `.git` directory containing previous commits, as they often expose author names and emails. 
- Initialize a fresh git repository:
  ```bash
  rm -rf .git
  git init
  git add .
  git commit -m "Initial commit for double-blind review"
  ```
- Ensure the `results/` and `checkpoints/` folders are appropriately excluded via `.gitignore` unless providing sample data.

## 3. Post-Acceptance
- The paper will un-anonymise the author listing and link to the original repository.
- Publish the Pip package `vit-bench` matching `pyproject.toml` specs under the verified maintainer profile.
