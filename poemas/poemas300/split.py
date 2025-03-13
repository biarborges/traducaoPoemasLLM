import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar o CSV com as colunas src_lang e tgt_lang
csv_path = "../traducaoPoemasLLM/poemas/poemas300/portugues_ingles_poems.csv"
df = pd.read_csv(csv_path)

# Dividir os dados em treinamento (80%) e temporário (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Dividir o temporário em validação (10%) e teste (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Salvar os conjuntos em arquivos CSV separados
train_df.to_csv("../traducaoPoemasLLM/poemas/poemas300/train/portugues_ingles_train.csv", index=False)
val_df.to_csv("../traducaoPoemasLLM/poemas/poemas300/validation/portugues_ingles_validation.csv", index=False)
test_df.to_csv("../traducaoPoemasLLM/poemas/poemas300/test/portugues_ingles_test.csv", index=False)

print("Arquivos salvos")