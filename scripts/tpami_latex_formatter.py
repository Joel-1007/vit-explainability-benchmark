import pandas as pd
import numpy as np

def generate_booktabs_table(df: pd.DataFrame, title: str, label: str) -> str:
    """
    Generate a strict IEEE TPAMI compliant LaTeX table using booktabs.
    Requires `\\usepackage{booktabs}` in the LaTeX preamble.
    """
    
    latex_str = []
    latex_str.append("\\begin{table*}[t]")
    latex_str.append(f"\\caption{{{title}}}")
    latex_str.append(f"\\label{{{label}}}")
    latex_str.append("\\centering")
    
    # Generate the column format (e.g., lcccccc)
    col_format = "l" + "c" * (len(df.columns))
    latex_str.append(f"\\begin{{tabular}}{{{col_format}}}")
    latex_str.append("\\toprule")
    
    # Headers
    headers = ["Method"] + list(df.columns)
    latex_str.append(" & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\")
    latex_str.append("\\midrule")
    
    # Data Rows
    # Assume the dataframe index contains the method names
    for index, row in df.iterrows():
        row_str = [str(index)]
        for val in row:
            # Highlight the best value in a column if needed (can be advanced explicitly)
            if isinstance(val, float):
                row_str.append(f"{val:.4f}")
            else:
                row_str.append(str(val))
        latex_str.append(" & ".join(row_str) + " \\\\")
    
    latex_str.append("\\bottomrule")
    latex_str.append("\\end{tabular}")
    latex_str.append("\\end{table*}")
    
    return "\n".join(latex_str)

if __name__ == "__main__":
    print("LaTeX Table Formatting Module for TPAMI. Use within analytical scripts phase 4.")
