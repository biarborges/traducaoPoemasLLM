import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from results.dicionario2 import equivalents

arquivo_entrada = "modelagemTopicos/results/ingles_portugues/topico3 - Copia.txt"
titulo = "EN-PT - Topic 3"
salvar_grafico = True
arquivo_grafico = "modelagemTopicos/results/ingles_portugues/ENPTtopico3sn.png"

# -------------------------------
# Normalizar e criar um mapeamento global de equivalentes
# -------------------------------
word_to_group = {}
for group_id, group_list in enumerate(equivalents):
    for word in group_list:
        word_norm = word.lower().strip()
        if word_norm not in word_to_group:
            word_to_group[word_norm] = group_id

# -------------------------------
# Lê topicos.txt (mantido igual)
# -------------------------------
data = {}
current_model = None

try:
    with open(arquivo_entrada, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^[A-Za-z].*:$', line):
                current_model = line[:-1].strip()
                data[current_model] = []
                continue
            # Dentro do laço de leitura do arquivo
            match = re.match(r'^(.*?):', line)

            match = re.match(r'^(.*?):', line)
            if match and current_model:
                word = match.group(1).strip().lower()
                data[current_model].append(word)

except FileNotFoundError:
    print(f"Erro: O arquivo '{arquivo_entrada}' não foi encontrado.")
    sys.exit()

# Limita a 10 palavras por tópico
max_words = 10
for model in data:
    data[model] = data[model][:max_words]

# Criar DataFrame
original_words = [f"Word{i+1}" for i in range(max_words)]
df = pd.DataFrame.from_dict(data, orient="index", columns=original_words)

# -------------------------------
# Contar frequência dos grupos de equivalentes
# -------------------------------
all_groups = []
for model in data:
    for word in data[model]:
        if pd.isna(word):
            continue
        
        
        word_norm = word.lower().strip() # Agora sim, normalize
        group_id = word_to_group.get(word_norm, None)


        # Usa o novo mapeamento
        group_id = word_to_group.get(word.lower().strip(), None)
        if group_id is not None:
            all_groups.append(group_id)

group_counts = Counter(all_groups)

# -------------------------------
# Preparar cores
# -------------------------------
unique_groups = list(dict.fromkeys(all_groups))
tab20_colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
if len(unique_groups) <= len(tab20_colors):
    rgb_palette = tab20_colors[:len(unique_groups)]
else:
    rgb_palette = [tuple(np.random.rand(3)) for _ in range(len(unique_groups))]

color_map = {group_id: rgb_palette[i] for i, group_id in enumerate(unique_groups)}

# -------------------------------
# Gráfico panorama colorido
# -------------------------------
color_matrix = []
for model in df.index:
    row_colors = []
    for word in df.loc[model]:
        if pd.isna(word):
            row_colors.append((1, 1, 1))  # branco para ausentes
            continue

        word_norm = word.lower().strip()
        group_id = word_to_group.get(word_norm, None)

        # Condição para colorir: a palavra deve pertencer a um grupo e esse grupo ter mais de uma ocorrência
        if group_id is None or group_counts[group_id] <= 1:
            row_colors.append((1, 1, 1))  # branco se não tiver equivalente ou for único
        else:
            row_colors.append(color_map[group_id])
    color_matrix.append(row_colors)

# -------------------------------
# Plot do gráfico (mantido igual)
# -------------------------------
fig, ax = plt.subplots(figsize=(12, len(df)*0.6))
ax.imshow(color_matrix, aspect='auto')

# Rótulos
ax.set_xticks(range(df.shape[1]))
ax.set_xticklabels(df.columns, rotation=45, ha='right')
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)

# Palavras dentro das células
for i, model in enumerate(df.index):
    for j, word in enumerate(df.loc[model]):
        if isinstance(word, str):
            ax.text(j, i, word, ha='center', va='center', color='black', fontsize=9)

# Grid
ax.set_xticks([x-0.5 for x in range(1, df.shape[1])], minor=True)
ax.set_yticks([y-0.5 for y in range(1, len(df.index))], minor=True)
ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
ax.tick_params(which="minor", length=0)

# Salvar ou mostrar gráfico
plt.title(titulo)
plt.tight_layout()

if salvar_grafico:
    plt.savefig(arquivo_grafico, dpi=300, bbox_inches="tight")
    plt.savefig(arquivo_grafico.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
else:
    plt.show()