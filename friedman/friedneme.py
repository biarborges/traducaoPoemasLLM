import pandas as pd
import numpy as np
import os
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
lg="portugues_ingles"
csv_path = "metricas_unificadas_portugues_ingles.csv"
use_only_full_models = True  # True: only models with 300 poems | False: include all (30 with fine-tuning)

metrics = ["bleu", "meteor", "bertscore", "bartscore"]
output_dir = "friedman_results_portugues_ingles" if use_only_full_models else "friedman_results_ft_portugues_ingles"
os.makedirs(output_dir, exist_ok=True)

# --- LOAD DATA ---
df = pd.read_csv(csv_path)

def get_columns(metric, include_ft=True):
    if include_ft:
        return [col for col in df.columns if col.startswith(metric)]
    else:
        return [col for col in df.columns if col.startswith(metric) and "_ft" not in col]

for metric in metrics:
    print(f"\n=== 🔎 METRIC: {metric.upper()} ===")

    columns = get_columns(metric, include_ft=not use_only_full_models)
    df_metric = df[columns].dropna()
    data = [df_metric[col].values for col in columns]

    print(f"\n📌 Comparing {len(columns)} models over {len(df_metric)} poems.")

    # --- FRIEDMAN TEST ---
    stat, p_value = friedmanchisquare(*data)
    friedman_summary = f"Friedman test for {metric.upper()}:\nStatistic = {stat:.4f}\nP-value = {p_value:.10e}\n"
    friedman_summary += "Significant difference found.\n" if p_value < 0.05 else "No significant difference found.\n"
    print(friedman_summary)

    # --- AVERAGE RANKING ---
    ranks = df_metric.rank(axis=1, method='average', ascending=False)
    mean_ranks = ranks.mean().sort_values()
    ranking_df = pd.DataFrame(mean_ranks, columns=["average_rank"])
    ranking_df.index.name = "model"
    ranking_df.reset_index(inplace=True)
    print("\n📈 Average ranking of the models:")
    print(ranking_df)

    # --- NEMENYI TEST ---
    nemenyi_result = sp.posthoc_nemenyi_friedman(df_metric.values)
    nemenyi_result.index = nemenyi_result.columns = columns
    print("\n📐 Nemenyi post-hoc test matrix:")
    print(nemenyi_result.round(4))

    # --- SAVE RESULTS ---
    metric_file = metric.lower()

    with open(os.path.join(output_dir, f"{metric_file}_friedman_{lg}.txt"), "w") as f:
        f.write(friedman_summary)

    ranking_df.to_csv(os.path.join(output_dir, f"{metric_file}_ranking_{lg}.csv"), index=False)
    nemenyi_result.to_csv(os.path.join(output_dir, f"{metric_file}_nemenyi_{lg}.csv"))

    # --- PLOT: AVERAGE RANKING BARPLOT ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=ranking_df.sort_values("average_rank", ascending=True),
        x="average_rank",
        y="model",
        hue="model",
        palette="crest",
        dodge=False,
        legend=False
    )
    fontsize=16
    # Aumentar tamanho das fontes
    plt.xlabel("Average Rank (lower = better)", fontsize=fontsize)
    plt.ylabel("Model", fontsize=fontsize)
    plt.title(f"Average Ranking of Models — {metric.upper()}", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_file}_ranking_{lg}.png"), dpi=300)
    plt.close()
    print(f"📸 Ranking plot saved to: {output_dir}/{metric_file}_ranking_{lg}.png")
