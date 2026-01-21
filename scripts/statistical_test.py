import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from config import RAW_RESULTS_PATH

def load_results(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def perform_statistical_analysis(data, metric="RMSE"):
    regressors = data.get("Regressors", {})

    # Extract metrics
    model_scores = {}
    for model_name, metrics in regressors.items():
        if metric in metrics:
            model_scores[model_name] = np.array(metrics[metric])
        
    means = {name: np.mean(scores) for name, scores in model_scores.items()}
    best_model = min(means, key=means.get)
    print(f"Best model based on mean {metric}: {best_model} with mean {means[best_model]:.4f}")

    # Pairwise Wilcoxon tests
    comparisons = []
    p_values = []
    other_models = [m for m in model_scores.keys() if m != best_model]

    for other_model in other_models:
        stat, p = wilcoxon(model_scores[best_model], model_scores[other_model])
        comparisons.append(other_model)
        p_values.append(p)
    
    # Correction of Holm-Bonferroni
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='holm')

    # DataFrame for results
    results_df = pd.DataFrame({
        "Model": comparisons,
        "p-value": p_values,
        "Corrected p-value": pvals_corrected,
        "Significant": ["Yes" if r else "No" for r in reject]
    })

    return best_model, results_df

def render_mpl_table(df, best_model_name, filename):
    if df is None: return

    fig, ax = plt.subplots(figsize=(10, 4)) # Tamaño de la imagen
    ax.axis('off')
    
    plt.title(f"Statistical Comparison vs. Best Model ({best_model_name})\nWilcoxon Test with Holm-Bonferroni Correction", 
              fontsize=14, pad=20, fontweight='bold')

    cell_text = []
    for row in df.itertuples(index=False):
        formatted_row = [
            row[0],
            f"{row[1]:.2e}" if row[1] < 0.001 else f"{row[1]:.4f}",
            f"{row[2]:.2e}" if row[2] < 0.001 else f"{row[2]:.4f}",
            row[3]
        ]
        cell_text.append(formatted_row)

    table = ax.table(cellText=cell_text,
                     colLabels=["Model Compared", "Original p-value", "Adj. p-value (Holm)", "Significant Diff?"],
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for (i, j), cell in table.get_celld().items():
        if i == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4C72B0')
            cell.set_edgecolor('white')
        else: # Datos
            cell.set_edgecolor('#dddddd')
            if i % 2 == 0:
                cell.set_facecolor('#f5f5f5')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Table saved to {filename}")
    plt.close()

def save_results_to_json(df, best_model_name, filename):
    if df is None: return
    
    output_data = {
        "best_model": best_model_name,
        "comparisons": df.to_dict(orient="records")
    }
    
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"JSON data saved to {filename}")

if __name__ == "__main__":
    data = load_results(RAW_RESULTS_PATH)
    best, df = perform_statistical_analysis(data, metric="RMSE")
    render_mpl_table(df, best, "./metrics/statistical_test/wilcoxon_holm_comparison.png")
    save_results_to_json(df, best, "./metrics/statistical_test/wilcoxon_holm_comparison.json")