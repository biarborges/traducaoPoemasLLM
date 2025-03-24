import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Carregar o modelo mBART para a tradução
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Mover o modelo para a GPU (se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Função para traduzir um poema (francês -> inglês)
def translate_poem(poem, src_lang="fr_XX", tgt_lang="en_XX", max_length=512):
    # Tokenizar o poema com truncamento para o comprimento máximo
    inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Mover os dados para a GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Definir as línguas de origem (francês) e destino (inglês)
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Gerar a tradução
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]  # Forçar o token de início para o idioma de destino
    )
    
    # Decodificar a tradução
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translation

# Carregar o CSV com os poemas (do francês)
file_path = "../poemas/poemas300/test/frances_ingles_test.csv"
df = pd.read_csv(file_path)

# Traduzir os poemas do francês para o inglês e adicionar ao DataFrame
df['translated_by_TA'] = df['original_poem'].apply(lambda poem: translate_poem(poem, src_lang="fr_XX", tgt_lang="en_XX"))

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/poemas300/mbart/frances_ingles_test_pretreinado_mbart.csv", index=False)

print("Tradução concluída e salva.")
