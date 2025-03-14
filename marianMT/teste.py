import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

# Caminhos dos arquivos
model_path = "/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-90"

input_file = os.path.abspath("../poemas/poemas300/test/frances_ingles_test.csv")
output_file = os.path.abspath("../poemas/poemas300/marianmt/frances_ingles_test_traducao_marianmt.csv")

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
    inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    translated_ids = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    translated_texts.append(translated_text)

# Adicionar a coluna com a tradução
df["translated_by_marian"] = translated_texts

# Reorganizar as colunas
df = df[["original_poem", "translated_poem", "translated_by_marian", "src_lang", "tgt_lang"]]

# Salvar no novo CSV
df.to_csv(output_file, index=False)

print(f"Tradução concluída! Arquivo salvo.")
