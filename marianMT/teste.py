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

# Traduzir cada poema em lotes
batch_size = 8  # Defina um tamanho de lote (ajuste conforme sua memória)
translated_texts = []

# Função para processar lotes
def translate_batch(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    translated_ids = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_ids]

# Traduzir em lotes
for i in range(0, len(df), batch_size):
    batch = df["original_poem"][i:i+batch_size].tolist()
    translated_batch = translate_batch(batch)
    translated_texts.extend(translated_batch)

# Adicionar a coluna com a tradução
df["translated_by_marian"] = translated_texts

# Reorganizar as colunas
df = df[["original_poem", "translated_poem", "translated_by_marian", "src_lang", "tgt_lang"]]

# Salvar no novo CSV
df.to_csv(output_file, index=False)

print(f"Tradução concluída! Arquivo salvo.")
