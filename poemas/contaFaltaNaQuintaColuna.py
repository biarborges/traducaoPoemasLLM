import pandas as pd

# Caminho do arquivo CSV
csv_path = "../traducaoPoemasLLM/poemas/googleTradutor/frances_ingles_poems_googleTradutor.csv"

# Carregar o CSV
df = pd.read_csv(csv_path)

# Contar quantas linhas possuem valores preenchidos na quinta coluna
num_filled = df["translated_by_TA"].notna().sum()

print(f"A quinta coluna possui {num_filled} linhas preenchidas.")
