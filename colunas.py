import pandas as pd

# Caminho do arquivo CSV
csv_path = "poemas/maritaca/portugues_ingles_test_maritaca_prompt2.csv" 

# Carrega o arquivo CSV
df = pd.read_csv(csv_path)

# Verifica se há valores vazios na coluna 'translated_by_TA'
linhas_vazias = df['translated_by_TA'].isna() | (df['translated_by_TA'].astype(str).str.strip() == '')

# Exibe as linhas onde a coluna está vazia
vazios = df[linhas_vazias]

# Mostra quantas linhas estão com a coluna vazia
print(f"Número de linhas com 'translated_by_TA' vazia: {len(vazios)}")

# (Opcional) Salvar essas linhas em um novo CSV
vazios.to_csv("poemas/maritaca/portugues_ingles_test_maritaca_prompt2_linhas_vazias.csv", index=False)