import torch
import os
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# Caminhos dos arquivos
model_path = "/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-150"
input_file = os.path.abspath("../poemas/poemas300/test/frances_ingles_test.csv")
output_file = os.path.abspath("../poemas/poemas300/marianmt/frances_ingles_test_traducao_marianmt3.csv")

# Verificar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer
model = MarianMTModel.from_pretrained(model_path).to(device)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Carregar dados do CSV
df = pd.read_csv(input_file)

# Traduzir cada poema
translated_texts = []
for poem in df["original_poem"]:
    # Tokenizar o poema original
    inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Gerar a tradução
    translated_ids = model.generate(**inputs)
    
    # Decodificar a tradução
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    # Adicionar a tradução à lista
    translated_texts.append(translated_text)

# Adicionar a coluna com a tradução ao DataFrame
df["translated_by_marian"] = translated_texts

# Salvar o novo CSV com as traduções
df.to_csv(output_file, index=False)

print(f"Tradução concluída! Arquivo salvo em {output_file}.")
