import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import time

time_start = time.time()

# Carregar o modelo mBART para francês -> inglês
model_name = "facebook/mbart-large-50-one-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Mover o modelo para a GPU (se disponível)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Função para traduzir um poema
def translate_poem(poem, src_lang="fr_XX", tgt_lang="en_XX", max_length=512):
    # Tokenizar o poema com truncamento para o comprimento máximo
    inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    # Mover os dados para a GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Definir as línguas de origem e destino corretamente
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
df['translated_by_TA'] = df['original_poem'].apply(lambda poem: translate_poem(poem, src_lang="fr_XX", tgt_lang="en_XX"))

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/poemas300/mbart/frances_ingles_test_pretreinado_mbart.csv", index=False)

print("Tradução concluída e salva.")

time_end = time.time()
elapsed_time = time_end - time_start
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
