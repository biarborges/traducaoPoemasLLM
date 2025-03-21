import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import time

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Modelos para tradução
model1 = "Helsinki-NLP/opus-mt-fr-en"  # Francês → Inglês
model2 = "Helsinki-NLP/opus-mt-tc-big-en-pt"  # Inglês → Português

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
    versos_traduzidos = traducao_intermediaria_texto.split("\n")
    traducao_final = []

    for verso in versos_traduzidos:
        encoded = tokenizer2(f">>pt<< {verso.strip()}", return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated_tokens = model2.generate(**encoded, max_length=512, num_beams=5)
        traducao_final.append(tokenizer2.decode(generated_tokens[0], skip_special_tokens=True))

    return "\n".join(traducao_intermediaria), "\n".join(traducao_final)

# Carregar o CSV com os poemas
df = pd.read_csv('../poemas/poemas300/test/frances_portugues_test.csv')

# Aplicar a tradução em duas etapas
df[['translated_fr_en', 'translated_fr_en_pt']] = df['original_poem'].apply(
    lambda x: traduzir_duas_etapas(x, tokenizer1, model1, tokenizer2, model2, device)
).apply(pd.Series)

# Salvar o CSV com as traduções
df.to_csv('../poemas/poemas300/marianmt/frances_portugues_test_pretreinado_marianmt.csv', index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
