import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import os
import torch
import time
from tqdm import tqdm

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do MarianMT
model_name = "Helsinki-NLP/opus-mt-ROMANCE-en"  
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Carregar CSV
input_file = os.path.abspath("../poemas/poemas300/frances_ingles_poems.csv")
output_file = os.path.abspath("../poemas/poemas300/marianmt/frances_ingles_traducao_marianmt.csv")

df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir textos em batch
def traduzir_textos(textos, batch_size=8):
    textos_com_prefixo = [">>en<< " + texto for texto in textos]  # Adiciona o prefixo de idioma
    
    traducoes = []
    for i in tqdm(range(0, len(textos_com_prefixo), batch_size), desc="Traduzindo", unit="batch"):
        batch = textos_com_prefixo[i:i+batch_size]
        
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Ativar FP16 para maior velocidade
                generated_tokens = model.generate(**encoded)
        
        batch_traduzido = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        traducoes.extend(batch_traduzido)
    
    return traducoes

# Aplicar a tradução para todos os poemas
df["translated_by_marian"] = traduzir_textos(df["original_poem"].tolist())

# Reorganizar as colunas na ordem desejada
df = df[["original_poem", "translated_poem", "translated_by_marian", "src_lang", "tgt_lang"]]

# Salvar em um novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
