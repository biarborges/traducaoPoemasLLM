import pandas as pd
from scikit_posthocs import critical_difference_diagram
import matplotlib.pyplot as plt

# Dados
df_ranking = pd.read_csv("friedman/friedman_results_portugues_ingles/bertscore_ranking_portugues_ingles.csv")
df_nemenyi = pd.read_csv("friedman/friedman_results_portugues_ingles/bertscore_nemenyi_portugues_ingles.csv", index_col=0)

ranks_dict = dict(zip(df_ranking["model"], df_ranking["average_rank"]))

# Tamanho reduzido e ajuste fino
fig, ax = plt.subplots(figsize=(10, 3.5))  # altura menor

critical_difference_diagram(ranks_dict, df_nemenyi, ax=ax)

# Ajuste do layout
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.05, right=0.95)

plt.tight_layout()
plt.savefig("cdd_bertscore_portugues_ingles_compacto.png", dpi=300)
plt.show()