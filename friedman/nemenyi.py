import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
lg="frances_ingles"
metric = "bertscore"  # Change to "bleu", "meteor", or "bartscore"
csv_file = "metricas_unificadas_frances_ingles.csv"
use_only_full_models = True  # True: only models without _ft (300 poems), False: includes _ft (30 poems)

output_dir = "nemenyi_results_comFT_frances_ingles"
os.makedirs(output_dir, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(csv_file)

# --- Function to select columns ---
def get_cols(metric, include_ft=True):
    if include_ft:
        return [col for col in df.columns if col.startswith(metric)]
    else:
        return [col for col in df.columns if col.startswith(metric) and "_ft" not in col]

# --- Run analysis ---
cols = get_cols(metric, include_ft=not use_only_full_models)
df_metric = df[cols].dropna()

if df_metric.shape[0] < 2:
    print(f"⚠️ Not enough data for metric {metric.upper()}")
    exit()

# Rankings
ranks = df_metric.rank(axis=1, method='average', ascending=False)
average_ranks = ranks.mean()

# Nemenyi test
nemenyi_result = sp.posthoc_nemenyi_friedman(df_metric.values)
models = [col.replace(f"{metric}_", "") for col in cols]
nemenyi_result.index = models
nemenyi_result.columns = models

# Save average rank CSV
ranking_df = pd.DataFrame({"model": models, "average_rank": average_ranks.values})
ranking_df = ranking_df.sort_values("average_rank")
ranking_df.to_csv(os.path.join(output_dir, f"{metric}_ranking.csv"), index=False)

# Save Nemenyi matrix CSV
nemenyi_result.to_csv(os.path.join(output_dir, f"{metric}_nemenyi_pvalues.csv"))

# --- Plot Nemenyi heatmap ---
plt.figure(figsize=(12, 10))

sns.heatmap(nemenyi_result,
            annot=True,
            fmt=".4f",
            cmap="coolwarm_r",  # colors: blue (low) to red (high)
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'p-value'},
            square=True,
            # mask=~mask_significant  # uncomment to show only p < 0.05
           )

plt.title(f"Nemenyi Test p-values Heatmap — {metric.upper()} (all values)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_path = os.path.join(output_dir, f"{metric}_nemenyi_heatmap_pvalues.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"✅ Results saved to folder: {output_dir}")
print(f"📊 Heatmap saved at: {heatmap_path}")
