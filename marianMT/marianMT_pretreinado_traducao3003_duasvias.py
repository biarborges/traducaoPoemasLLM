#esse aqui é pros 30 pre treinados.
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import time

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Modelos para tradução
#model1 = "Helsinki-NLP/opus-mt-ROMANCE-en"  # Francês → Inglês
#model2 = "Helsinki-NLP/opus-mt-en-pt"  # Inglês → Português

model1 = "/home/ubuntu/finetuning_pt_en/checkpoint-90"  # Francês → Inglês
model2 = "/home/ubuntu/finetuning_ing_fr/checkpoint-90"  # Inglês → Português

# Carregar tokenizers e modelos
tokenizer1 = MarianTokenizer.from_pretrained(model1)
model1 = MarianMTModel.from_pretrained(model1).to(device)

tokenizer2 = MarianTokenizer.from_pretrained(model2)
model2 = MarianMTModel.from_pretrained(model2).to(device)

# Função para traduzir poema em duas etapas
def traduzir_duas_etapas(poema, tokenizer1, model1, tokenizer2, model2, device):
    versos = poema.split("\n")
    traducao_intermediaria = []

    # Primeira etapa: Francês → Inglês
    for verso in versos:
        encoded = tokenizer1(f">>en<< {verso.strip()}", return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated_tokens = model1.generate(**encoded, max_length=512, num_beams=5)
        traducao_intermediaria.append(tokenizer1.decode(generated_tokens[0], skip_special_tokens=True))

    traducao_intermediaria_texto = "\n".join(traducao_intermediaria)

    # Segunda etapa: Inglês → Português
    traducao_final = []
    for verso in traducao_intermediaria_texto.split("\n"):
        encoded = tokenizer2(f">>fr<< {verso.strip()}", return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated_tokens = model2.generate(**encoded, max_length=512, num_beams=5)
        traducao_final.append(tokenizer2.decode(generated_tokens[0], skip_special_tokens=True))

    return "\n".join(traducao_final)

# Carregar o CSV com os poemas
df = pd.read_csv('../poemas/poemas300/test/portugues_frances_test.csv')

# Aplicar a tradução e salvar apenas a versão final
df['translated_by_marian'] = df['original_poem'].apply(
    lambda x: traduzir_duas_etapas(x, tokenizer1, model1, tokenizer2, model2, device)
)

# Salvar o CSV com apenas a tradução final
df.to_csv('../poemas/poemas300/marianmt/portugues_frances_test_finetuning_marianmt.csv', index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
