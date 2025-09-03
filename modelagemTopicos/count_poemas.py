import pandas as pd

# Caminho para o arquivo CSV com os tópicos
CAMINHO_CSV = "modelagemTopicos/results/ingles_frances/googleTradutor/ingles_frances_poems_googleTradutor.csv"

# Lê o arquivo CSV
df = pd.read_csv(CAMINHO_CSV)

# Conta o total de poemas
total_poemas = len(df)

# Exibe os resultados
print("Quantidade total de poemas:", total_poemas)