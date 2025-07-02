import pandas as pd

# Caminhos dos arquivos
arquivo_topicos = "results/ingles_portugues/original/poemas_com_topicos_original.csv"
arquivo_traducao = "results/ingles_portugues/maritacaPrompt2/ingles_portugues_poems_maritaca_Prompt2.csv"
saida_arquivo = "results/ingles_portugues/maritacaPrompt2/poemas_com_topicos_maritacaPrompt2.csv"

# Lê os dois arquivos CSV
df_topicos = pd.read_csv(arquivo_topicos)
df_traducao = pd.read_csv(arquivo_traducao)

# Faz o merge com base na coluna 'original_poem'
df_merged = pd.merge(df_traducao, df_topicos[['original_poem', 'topic']], on='original_poem', how='left')

# Salva o novo arquivo com os tópicos adicionados
df_merged.to_csv(saida_arquivo, index=False)

print(f"✅ Arquivo salvo com os tópicos em: {saida_arquivo}")
