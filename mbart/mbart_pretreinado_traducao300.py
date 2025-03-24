import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do mBART
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Função para traduzir poema
def traduzir_poema(poema, src_lang="en_XX", tgt_lang="fr_XX"):
    # Configurar a língua de origem e destino no tokenizer
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Tokenizar o verso
    encoded = tokenizer(poema.strip(), return_tensors="pt", truncation=True, padding=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU

    # Gerar tradução
    with torch.no_grad():
        generated_tokens = model.generate(**encoded, max_length=512, num_beams=5)

    traducao = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return traducao

# Carregar o arquivo CSV com os poemas
file_path = "../poemas/poemas300/test/ingles_frances_test.csv"  # Altere o caminho do arquivo conforme necessário
df = pd.read_csv(file_path)

# Traduzir os poemas e adicionar à nova coluna 'translated_by_TA'
df['translated_by_TA'] = df['original_poem'].apply(lambda poem: traduzir_poema(poem, src_lang="en_XX", tgt_lang="fr_XX"))

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/poemas300/mbart/ingles_frances_test_pretreinado_mbart.csv", index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")