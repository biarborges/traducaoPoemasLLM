import pandas as pd
from scikit_posthocs import critical_difference_diagram
import matplotlib.pyplot as plt

# Dados
df_ranking = pd.read_csv("friedman/friedman_results_frances_portugues/bertscore_ranking_frances_portugues.csv")
df_nemenyi = pd.read_csv("friedman/friedman_results_frances_portugues/bertscore_nemenyi_frances_portugues.csv", index_col=0)

ranks_dict = dict(zip(df_ranking["model"], df_ranking["average_rank"]))

# Figura
fig, ax = plt.subplots(figsize=(10, 3.5))

# Gera o diagrama com fonte aumentada e negrito
critical_difference_diagram(
    ranks_dict,
    df_nemenyi,
    ax=ax,
    label_props={'fontsize': 20, 'fontweight': 'bold'},  # aqui que muda o texto!
    marker_props={'s': 60},  # aumenta tamanho dos marcadores (opcional)
)

# Ajusta layout
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.05, right=0.95)
plt.tight_layout()
plt.savefig("cd_plotFRPT.png", dpi=300)
plt.show()
