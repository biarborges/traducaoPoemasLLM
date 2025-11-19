import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Caminho do CSV original
input_file = r"C:\Users\biarb\OneDrive\UFU\Mestrado\Dissertacao\traducaoPoemasLLM\poemas\portugues_frances_poems.csv"

# Lê o CSV
df = pd.read_csv(input_file)

# Divide 60% treino e 40% restante
df_train, df_rest = train_test_split(df, test_size=0.4, random_state=42)

# Divide o restante em 50% validação e 50% teste (cada 20% do total)
df_validation, df_test = train_test_split(df_rest, test_size=0.5, random_state=42)

# Caminhos dos arquivos de saída
base_path = os.path.dirname(input_file)
train_file = os.path.join(base_path, "portugues_frances_train.csv")
validation_file = os.path.join(base_path, "portugues_frances_validation.csv")
test_file = os.path.join(base_path, "portugues_frances_test.csv")

# Salva os CSVs
df_train.to_csv(train_file, index=False)
df_validation.to_csv(validation_file, index=False)
df_test.to_csv(test_file, index=False)

print("ok.")
