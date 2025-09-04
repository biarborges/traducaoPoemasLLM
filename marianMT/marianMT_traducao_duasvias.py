import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import time

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Modelos para tradução
#model1 = "Helsinki-NLP/opus-mt-ROMANCE-en"  
#model2 = "Helsinki-NLP/opus-mt-en-pt"  

#model1 = "/home/ubuntu/finetuning_fr_en/checkpoint-1013"
#model2 = "/home/ubuntu/finetuning_en_pt/checkpoint-2433"

tokenizer1 = MarianTokenizer.from_pretrained("/home/ubuntu/finetuning_fr_en")
model1 = MarianMTModel.from_pretrained("/home/ubuntu/finetuning_fr_en/checkpoint-1013").to(device)


tokenizer2 = MarianTokenizer.from_pretrained("/home/ubuntu/finetuning_en_pt")
model2 = MarianMTModel.from_pretrained("/home/ubuntu/finetuning_en_pt/checkpoint-2433").to(device)


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
        encoded = tokenizer2(f">>pt<< {verso.strip()}", return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated_tokens = model2.generate(**encoded, max_length=512, num_beams=5)
        traducao_final.append(tokenizer2.decode(generated_tokens[0], skip_special_tokens=True))

    return "\n".join(traducao_final)

# Carregar o CSV com os poemas
df = pd.read_csv('../poemas/frances_portugues_poems.csv')

# Adicionar barra de progresso com tqdm
tqdm.pandas(desc="Traduzindo poemas em duas etapas")

# Aplicar a tradução com progresso
df['translated_by_TA'] = df['original_poem'].progress_apply(
    lambda x: traduzir_duas_etapas(x, tokenizer1, model1, tokenizer2, model2, device)
)

# Salvar o CSV com apenas a tradução final
df.to_csv('../poemas/marianmt/finetuning_musics/frances_portugues.csv', index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
