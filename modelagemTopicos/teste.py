import pandas as pd
import os

# Caminhos de entrada
caminho_original = "results/portugues_ingles/original/poemas_com_topicos_original.csv"  # com a coluna 'topic'
caminho_automatico = "results/portugues_ingles/maritacaPrompt2/portugues_ingles_poems_maritaca_Prompt2.csv"  #arquivo sem o topic
saida_arquivo = "results/portugues_ingles/maritacaPrompt2/poemas_com_topicos_maritacaPrompt2.csv"

# Carrega os DataFrames
df_original = pd.read_csv(caminho_original)
df_auto = pd.read_csv(caminho_automatico)

# Junta os dados pelo 'original_poem', mas não sobrescreve os arquivos originais
df_com_topico = pd.merge(df_auto, df_original[['original_poem', 'topic']], on='original_poem', how='left')

# Salva o resultado em outro local
df_com_topico.to_csv(saida_arquivo, index=False)
print(f"✅ Arquivo salvo em: {saida_arquivo}")
