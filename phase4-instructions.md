# Phase 4 Instructions: Analysis, Ablations & Theoretical Grounding

This document outlines the operational process to execute the scripts created for Phase 4 of the project. These analytics convert the raw metric results matrix computed in Phase 3 into theoretical insights, empirical findings, and paper-ready figures and tables.

## Prerequisites
Ensure that all Phase 3 metric evaluation logic has successfully concluded and that an aggregated Results CSV containing the benchmark scores has been finalized. Ensure the Phase 4 output directory `results/phase4/` is present.

### Environment Setup
Phase 4 analytics utilize standard PyData libraries. Ensure your environment contains at least the following:
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`

Activate the Python virtual environment containing these libraries:
```bash
# If using a virtual environment on Windows:
.venv\Scripts\activate
```

---

## Running the Analyses

There are three analysis scripts in the `scripts/` directory to facilitate Tasks 4.1, 4.2, and 4.3. Run these scripts from the root directory of the project.

### 1. Inter-Metric Correlation Analysis (Task 4.1)
This step explores whether the metrics measure orthogonal properties of explainers by evaluating Spearman rank correlation pairings. It also contains scaffolding for PCA/Factor Analysis.

**Execution:**
```bash
python scripts/phase4_correlation_analysis.py --results_csv path/to/aggregated_results.csv --output_dir results/phase4
```
**Expected Output:**
- `results/phase4/inter_metric_correlations.csv`
- `results/phase4/inter_metric_heatmap.pdf`
- `results/phase4/factor_analysis_results.csv`

### 2. Task-Metric Interaction Analysis (Task 4.2)
Computes explainer ranking concordance using Kendall's tau correlation across task domains (datasets). This helps answer whether explanation quality is invariant across datasets.

**Execution:**
```bash
python scripts/phase4_interaction_analysis.py --results_csv path/to/aggregated_results.csv --output_dir results/phase4
```
**Expected Output:**
- `results/phase4/explainer_rankings.csv`
- `results/phase4/dataset_concordance.csv`

### 3. Ablation Studies (Task 4.3)
Quantifies systematic behavior via controlled variables. Evaluates Token Resolution (A1), Layer Depth (A2), Masking Strategy (A3), and Pre-training Objective (A4). Computes relevant effect sizes via Cohen's d.

**Execution:**
```bash
python scripts/phase4_ablations.py --output_dir results/phase4
```
**Expected Output:**
- Individual ablation tables and logged execution paths (update dataloading prior to scaling up execution).

---

## Updating Project Documentation

The structural placeholders for Phase 4 have already been added to the documentation files. After running the analysis scripts, you MUST NOT reconstruct tables or headers. **Only update the placeholders with raw result metrics, generated figures, and inferential observations.**

### 1. `BENCHMARK.md` Updates
- Locate the `Phase 4 — Analysis, Ablations & Theoretical Grounding` section at the bottom of the file.
- Replace the `[PLACEHOLDER]` tags with actual data:
  - Inject the Kendall tau and Spearman rank metrics directly to justify metric orthogonality.
  - Fill the Ablation Studies table with computed Cohen's d effect sizes.
  - Paste the rendered Markdown for the practitioner's decision tree directly where prompted.
  - Detail any analytical inferences regarding metric stability or explanation quality degradation.

### 2. `README.md` Updates
- Locate the `## 📊 Phase 4: Analysis & Findings` section.
- Replace the placeholder bullets with a concise 1-2 sentence summary of the major conclusions discovered (e.g. the most stable metric, the strongest performing ablation strategy).

> [!NOTE]
> All analytical outputs meant for manuscript submission (e.g., TPAMI) must be formatted cleanly within tabular formats. Consult Section 5.2 of the `implementation_guide.md` for specific publication styling specifics (`pdf` vectors only, `booktabs` for tables). Make sure color maps are color-blind-safe (e.g., `coolwarm`, `viridis`).
