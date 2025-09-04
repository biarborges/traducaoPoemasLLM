import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import time

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do MarianMT
model_name = "/home/ubuntu/finetuning_en_pt/checkpoint-1023"
#model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Função para traduzir poema
def traduzir_poema(poema, tokenizer, model, device):
    versos = poema.split("\n")
    traducao_completa = []

    # Traduzir verso por verso
    for verso in versos:
        texto_com_prefixo = f">>pt<< {verso.strip()}"  # Adicionar prefixo da língua
        encoded = tokenizer(texto_com_prefixo, return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU

        with torch.no_grad():
            generated_tokens = model.generate(**encoded, max_length=512, num_beams=5)

        traducao = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        traducao_completa.append(traducao)

    return "\n".join(traducao_completa)

# Carregar o CSV com os poemas
#df = pd.read_csv('../poemas/test/ingles_portugues_test.csv')
df = pd.read_csv('../poemas/ingles_portugues_poems.csv')

# Usar tqdm para mostrar o progresso da tradução
tqdm.pandas(desc="Traduzindo poemas")

# Adicionar a coluna para as traduções com barra de progresso
df['translated_by_TA'] = df['original_poem'].progress_apply(lambda x: traduzir_poema(x, tokenizer, model, device))

# Salvar o CSV com a tradução
df.to_csv('../poemas/marianmt/finetuning_musics/ingles_portugues.csv', index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
