import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Definir idiomas
SRC_LANG = "fr_XX"
TGT_LANG = "en_XX"

# Carregar CSV
input_file = os.path.abspath("../poemas/poemas300/frances_ingles_poems.csv")
output_file = os.path.abspath("../poemas/poemas300/test/frances_ingles_poems_traducao_mbart.csv")

df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir um poema
def traduzir_texto(texto):
    tokenizer.src_lang = SRC_LANG
    # Dividir o poema em estrofes (ou versos) usando o separador de linha
    estrofes = texto.split('\n')

    traducao_completa = []
    # Barra de progresso para estrofes dentro de cada poema
    for estrofe in tqdm(estrofes, desc="Traduzindo estrofes", unit="estrofe"):
        # Tokenizar e traduzir cada estrofe separadamente
        encoded = tokenizer(estrofe, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover os dados para a GPU
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG])
        traducao_completa.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))

    # Juntar as estrofes traduzidas de volta
    return '\n'.join(traducao_completa)

# Função para traduzir todos os poemas com contador global
def traduzir_com_contador(row, index, total_poemas):
    print(f"Traduzindo poema {index+1}/{total_poemas}")
    return traduzir_texto(row["original_poem"])

# Barra de progresso para poemas
tqdm.pandas(desc="Traduzindo poemas", total=len(df))

# Aplicar a tradução para cada linha do CSV com barra de progresso para os poemas
df["translated_by_mbart"] = [traduzir_com_contador(row, index, len(df)) for index, row in tqdm(df.iterrows(), total=len(df), desc="Traduzindo poemas")]

# Reorganizar as colunas na ordem desejada
df = df[["original_poem", "translated_poem", "translated_by_mbart", "src_lang", "tgt_lang"]]

# Salvar em um novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")