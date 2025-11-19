import pandas as pd

# Caminho para o CSV (usar \\ ou /)
csv_file = "C:/Users/biarb/OneDrive/UFU/Mestrado/Dissertacao/traducaoPoemasLLM/poemas/validation/portugues_frances_validation.csv"

# Lendo o CSV
df = pd.read_csv(csv_file)

# Contando o número de objetos (linhas)
num_objetos = len(df)

print(f"O CSV contém {num_objetos} objetos.")
