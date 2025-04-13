import pandas as pd

# Carregar os dois arquivos CSV
df1 = pd.read_csv('../traducaoPoemasLLM/poemas/maritaca/portugues_ingles_test_maritaca_prompt2.csv')
df2 = pd.read_csv('../traducaoPoemasLLM/poemas/maritaca/portugues_ingles_test_maritaca_prompt2_linhas_vazias_preenchidas.csv')

# Criar um dicionário a partir do df2 usando como chave a translated_poem
# Isso nos permitirá buscar rapidamente o valor correspondente
translation_dict = df2.set_index(['translated_poem'])['translated_by_TA'].to_dict()

# Função para preencher os valores faltantes em df1
def fill_missing(row):
    # Se translated_by_TA estiver vazio ou NaN
    if pd.isna(row['translated_by_TA']) or row['translated_by_TA'] == "":
        key = (row['translated_poem'])
        # Buscar no dicionário
        return translation_dict.get(key, row['translated_by_TA'])  # Se não encontrar, mantém o original
    return row['translated_by_TA']

# Aplicar a função para preencher os valores
df1['translated_by_TA'] = df1.apply(fill_missing, axis=1)

# Salvar o resultado (pode ser um novo arquivo ou sobrescrever o original)
df1.to_csv('../traducaoPoemasLLM/poemas/maritaca/portugues_ingles_test_maritaca_prompt2_completo.csv', index=False)

print("Concluído! O arquivo foi salvo como '../traducaoPoemasLLM/poemas/maritaca/ingles_portugues_test_maritaca_prompt2_completo.csv'")