import pandas as pd
import os

# Caminho do arquivo CSV original
csv_path = "../traducaoPoemasLLM/poemas/frances_ingles_poems.csv"

# Caminho da pasta de saída
output_folder = "../traducaoPoemasLLM/poemas/de10em10"

# Garantir que a pasta de saída existe
os.makedirs(output_folder, exist_ok=True)

# Carregar o CSV original
df = pd.read_csv(csv_path)

# Número de linhas por novo CSV
lines_per_file = 10

# Número total de novos arquivos CSV
num_files = len(df) // lines_per_file

# Criar os novos CSVs
for i in range(num_files):
    start_row = i * lines_per_file
    end_row = start_row + lines_per_file
    new_df = df.iloc[start_row:end_row]
    
    # Nome do novo arquivo CSV com o caminho da pasta de saída
    new_csv_path = os.path.join(output_folder, f"frances_ingles_poems_{i+1}.csv")
    
    # Salvar o novo CSV
    new_df.to_csv(new_csv_path, index=False)
    print(f"Novo arquivo CSV salvo como: {new_csv_path}")
