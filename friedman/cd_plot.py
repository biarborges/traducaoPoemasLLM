import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- 1. Leitura dos Arquivos ---
try:
    df_ranking = pd.read_csv("friedman/friedman_results_frances_ingles/bertscore_ranking_frances_ingles.csv")
    df_nemenyi = pd.read_csv("friedman/friedman_results_frances_ingles/bertscore_nemenyi_frances_ingles.csv", index_col=0)
    # Não precisamos ler o arquivo friedman.txt para o plot, apenas para informação
except FileNotFoundError:
    print("Erro: Certifique-se de que 'bertscore_ranking_frances_ingles.csv' e 'bertscore_nemenyi_frances_ingles.csv' estão no mesmo diretório do script.")
    exit() # Sai do script se os arquivos não forem encontrados

# --- 2. Preparação dos Dados ---
# Ordenar os modelos pelo rank médio para o plot
df_ranking_sorted = df_ranking.sort_values(by='average_rank').reset_index(drop=True)
models = df_ranking_sorted['model'].tolist()
ranks = df_ranking_sorted['average_rank'].tolist()

# Nível de significância para o teste de Nemenyi
alpha = 0.05

# --- 3. Lógica para Agrupamento (Identificação de Componentes Conectados) ---
# Função auxiliar para combinar grupos sobrepostos
def combine_overlapping_sets(list_of_sets):
    """Combina conjuntos (grupos) em uma lista que possuem elementos em comum."""
    result = []
    for s in list_of_sets:
        added_to_existing = False
        for i, existing_s in enumerate(result):
            if not existing_s.isdisjoint(s): # Se há intersecção
                result[i] = existing_s.union(s) # Combina os conjuntos
                added_to_existing = True
                break
        if not added_to_existing:
            result.append(s)
    # Repetir até não haver mais combinações possíveis
    if len(result) < len(list_of_sets):
        return combine_overlapping_sets(result)
    return result

# Lista para armazenar os grupos de modelos inicialmente identificados
initial_groups = []
# Conjunto para controlar quais modelos já foram adicionados a um grupo maior
grouped_models_tracker = set()

# Iterar sobre todos os modelos para encontrar seus "não-diferentes"
for i, model_i in enumerate(models):
    if model_i not in grouped_models_tracker:
        current_group_members = {model_i}
        # Encontrar todos os modelos que não são significativamente diferentes de model_i
        for j, model_j in enumerate(models):
            if i != j: # Não comparar o modelo consigo mesmo
                try:
                    # Garantir que os nomes das colunas/índices estejam corretos no df_nemenyi
                    p_value = df_nemenyi.loc[model_i, model_j]
                except KeyError:
                    # Se houver inconsistência nos nomes, tenta o inverso
                    p_value = df_nemenyi.loc[model_j, model_i]

                if p_value > alpha:
                    current_group_members.add(model_j)
        
        if len(current_group_members) > 1:
            initial_groups.append(current_group_members)
            # Não adicionar ao tracker aqui, pois a combinação final cuidará disso
            # grouped_models_tracker.update(current_group_members) # Isso causaria problemas na combinação

# Combina os grupos iniciais que se sobrepõem para formar os grupos finais
final_groups_to_plot = combine_overlapping_sets(initial_groups)
# Filtra para garantir que apenas grupos com mais de um membro sejam plotados como linhas
final_groups_to_plot = [list(group) for group in final_groups_to_plot if len(group) > 1]


# --- 4. Plotagem do Diagrama de Diferença Crítica ---
plt.figure(figsize=(10, 5))
y_pos_main_line = 0.5 # Posição da linha horizontal principal dos ranks

# Mapeia nomes de modelos para ranks para fácil acesso
model_rank_map = dict(zip(models, ranks))

# Define as cores para cada modelo
colors = plt.cm.get_cmap('tab10', len(models))

# Define os limites do eixo X
min_rank = min(ranks)
max_rank = max(ranks)
x_padding = (max_rank - min_rank) * 0.1 # Adiciona um pouco de espaço nas bordas
plt.xlim(min_rank - x_padding, max_rank + x_padding)

# Plotar os pontos e rótulos dos modelos
for i, model_name in enumerate(models):
    rank_value = model_rank_map[model_name]
    plt.plot(rank_value, y_pos_main_line, 'o', markersize=8, color=colors(i))
    # Ajusta a posição Y do texto para evitar sobreposição
    plt.text(rank_value, y_pos_main_line - 0.01 - (i % 4) * 0.04,
             f"{model_name.replace('bertscore_', '').replace('_', ' ')} ({rank_value:.2f})",
             verticalalignment='top', horizontalalignment='center',
             fontsize=12, color=colors(i))

# Desenhar a linha horizontal principal que representa o eixo dos ranks
plt.axhline(y_pos_main_line, color='black', linestyle='-', linewidth=1.5)

# Desenhar as linhas de agrupamento (barras horizontais e verticais)
y_offset_step = 0.05 # Incremento para cada linha de grupo
current_y_line_level = y_pos_main_line + 0.1 # Nível inicial das linhas de grupo

for group in final_groups_to_plot:
    # Obter os ranks dos modelos que pertencem a este grupo
    group_ranks = [model_rank_map[m] for m in group]
    min_group_rank = min(group_ranks)
    max_group_rank = max(group_ranks)

    # Desenhar a linha horizontal do grupo
    plt.hlines(current_y_line_level, min_group_rank, max_group_rank,
               colors='black', linewidth=3)

    # Desenhar as linhas verticais para conectar os modelos do grupo à linha principal
    for rank_val in group_ranks:
        plt.vlines(rank_val, y_pos_main_line, current_y_line_level,
                   colors='black', linestyle='-', linewidth=1)
    
    current_y_line_level += y_offset_step # Avança para o próximo nível Y para o próximo grupo

# --- 5. Configurações Finais do Plot ---
plt.title("BERTScore Critical Difference Diagram (French-English)", fontsize=14)
plt.xlabel("Average Rank", fontsize=14)
plt.yticks([]) # Ocultar o eixo Y, pois não tem significado aqui
plt.ylim(0.2, current_y_line_level + 0.05) # Ajustar os limites Y para que tudo caiba

tick_step = 0.5 # Experimente com 0.5, 0.25, 0.1, etc. para ver o que funciona melhor
plt.xticks(np.arange(np.floor(min_rank), np.ceil(max_rank) + tick_step, tick_step))


# Adicionar linhas de grade suaves no eixo X para auxiliar a leitura dos ranks
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout() # Ajusta automaticamente os parâmetros do subplot para que caibam na figura
plt.show()