import pandas as pd
import os

# Caminho do arquivo CSV
input_file = os.path.abspath("../traducaoPoemasLLM/poemas/poemas300/test/frances_ingles_test.csv")
output_file = os.path.abspath("../traducaoPoemasLLM/poemas/poemas300/test/frances_ingles_test2.csv")

# Carregar os dados do CSV
df = pd.read_csv(input_file)

# Função para remover quebras de linha e juntar o texto em um único parágrafo
def remove_newlines(text):
    return " ".join(text.splitlines())

# Aplicar a função para remover quebras de linha na coluna 'original_poem'
df['original_poem'] = df['original_poem'].apply(remove_newlines)

# Salvar os dados no novo CSV
df.to_csv(output_file, index=False)

print("Quebras de linha removidas e novo arquivo salvo.")
