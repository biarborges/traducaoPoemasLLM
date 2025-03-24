import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Carregar o modelo mBART para francês -> inglês
model_name = "facebook/mbart-large-50-one-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Função para traduzir um poema
def translate_poem(poem, src_lang="fr", tgt_lang="en"):
    # Tokenizar o poema
    inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True)

    # Definir as línguas de origem e destino
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Gerar a tradução
    translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translation

# Carregar o CSV com os poemas
file_path = "../poemas/poemas300/test/frances_ingles_test.csv"
df = pd.read_csv(file_path)

# Traduzir os poemas do francês para o inglês e adicionar ao DataFrame
df['translated_by_TA'] = df['original_poem'].apply(lambda poem: translate_poem(poem, src_lang="fr", tgt_lang="en"))

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/poemas300/test/frances_ingles_test_pretreinado_mbart.csv", index=False)

# Verificar as primeiras traduções
print(df[['original_poem', 'translated_by_TA']].head())
