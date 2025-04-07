import pandas as pd

# Dicionário para mapear pares de línguas para códigos de idioma do mBART
language_pairs = {
    "frances_ingles": ("fr_XX", "en_XX"),
    "ingles_frances": ("en_XX", "fr_XX"),
    "portugues_ingles": ("pt_XX", "en_XX"),
    "ingles_portugues": ("en_XX", "pt_XX"),
    "frances_portugues": ("fr_XX", "pt_XX"),
    "portugues_frances": ("pt_XX", "fr_XX"),
}

# Função para adicionar as colunas src_lang e tgt_lang
def add_language_columns(csv_path, src_lang, tgt_lang):
    # Carregar o CSV
    df = pd.read_csv(csv_path)
    
    # Adicionar as colunas src_lang e tgt_lang
    df["src_lang"] = src_lang
    df["tgt_lang"] = tgt_lang
    
    # Salvar o CSV atualizado
    new_csv_path = ("portugues_ingles_poems.csv")
    df.to_csv(new_csv_path, index=False)
    print(f"Arquivo atualizado salvo em: {new_csv_path}")

# Exemplo de uso
csv_path = "portugues_ingles_poems_data_normalized.csv"
src_lang, tgt_lang = language_pairs["portugues_ingles"]
add_language_columns(csv_path, src_lang, tgt_lang)