import pandas as pd

# Caminho para o arquivo CSV com os tópicos
CAMINHO_CSV = "../modelagemTopicos/results/portugues_ingles_original/traducaoReferencia/poemas_com_topicos.csv"

# Lê o arquivo CSV
df = pd.read_csv(CAMINHO_CSV)

# Conta a quantidade de poemas por tópico
contagem_por_topico = df["topic"].value_counts().sort_index()

# Exibe a contagem por tópico
print("Quantidade de poemas por tópico:")
print(contagem_por_topico)
