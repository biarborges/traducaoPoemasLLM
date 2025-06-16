import pandas as pd

# Caminhos dos arquivos
arquivo_topicos = "modelagemTopicos/results/portugues_ingles_original/poemas_com_topicos.csv"
arquivo_traducao = "modelagemTopicos/googleTradutor/portugues_ingles_poems_googleTradutor.csv"
saida_arquivo = "modelagemTopicos/results/portugues_ingles_original/googleTradutor/poemas_com_topicos.csv"

# Lê os dois arquivos CSV
df_topicos = pd.read_csv(arquivo_topicos)
df_traducao = pd.read_csv(arquivo_traducao)

# Faz o merge com base na coluna 'original_poem'
df_merged = pd.merge(df_traducao, df_topicos[['original_poem', 'topic']], on='original_poem', how='left')

# Salva o novo arquivo com os tópicos adicionados
df_merged.to_csv(saida_arquivo, index=False)

print(f"✅ Arquivo salvo com os tópicos em: {saida_arquivo}")
