import pandas as pd
import os

# Caminho do arquivo CSV com os tópicos
CAMINHO_CSV = "results/ingles_frances/reference/poemas_com_topicos_reference.csv"
DIRETORIO_SAIDA = "results/ingles_frances/reference"

COLUNA_TOPICO = "topic"

# Lê o CSV
df = pd.read_csv(CAMINHO_CSV)

# Remove outliers (tópico -1)
df = df[df[COLUNA_TOPICO] != -1]

# Cria diretório de saída
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

# Agrupa e salva por tópico
for topico, grupo in df.groupby(COLUNA_TOPICO):
    caminho_saida = os.path.join(DIRETORIO_SAIDA, f"topico_{topico}.csv")
    grupo.to_csv(caminho_saida, index=False)

print(f"✅ Poemas separados por tópico salvos em: {DIRETORIO_SAIDA}")
