import pandas as pd

# Caminho do CSV original
arquivo_original = "metricas_unificadas_ingles_portugues.csv"

# Nome do novo CSV com apenas as colunas de BERTScore
arquivo_saida = "bertscore_ingles_portugues.csv"

# Carrega o CSV original
df = pd.read_csv(arquivo_original)

# Seleciona somente a coluna "poema_id" + colunas que começam com "bertscore_"
colunas_bertscore = ["poema_id"] + [col for col in df.columns if col.startswith("bertscore_")]

# Cria o novo DataFrame apenas com essas colunas
df_bertscore = df[colunas_bertscore]

# Salva o novo CSV
df_bertscore.to_csv(arquivo_saida, index=False)

print(f"✅ Arquivo salvo como: {arquivo_saida}")
