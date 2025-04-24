#iual ao 300, só q forçando o bos_token_id para a lingua tgt para garantir a lingua.

import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm  # <- importar tqdm

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do mBART
model_name = "/home/ubuntu/finetuning_fr_pt/checkpoint-297"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

tokenizer.model_max_length = 1024  # <- força para 1024 tokens

# Função para traduzir poema
def traduzir_poema_em_partes(poema, src_lang="fr_XX", tgt_lang="pt_XX"):
    if not isinstance(poema, str) or poema.strip() == "":
        return ""
    
    partes = poema.split("\n")  # Divide o poema por estrofes (quebra de linha)
    traducao = []

    for parte in partes:
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

        encoded = tokenizer(parte.strip(), return_tensors="pt", truncation=True, padding=True, max_length=1024)
        encoded = {key: value.to(device) for key, value in encoded.items()}

        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                max_length=1024,
                num_beams=5,
                forced_bos_token_id=forced_bos_token_id
            )

        traducao.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
    
    return "\n".join(traducao)  # Junta as estrofes traduzidas

# Carregar o arquivo CSV com os poemas
file_path = "../poemas/test/frances_portugues_test.csv"
df = pd.read_csv(file_path)

# Traduzir os poemas com barra de progresso
translated_poems = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Traduzindo poemas"):
    translated_poems.append(traduzir_poema_em_partes(row['original_poem'], src_lang=row['src_lang'], tgt_lang=row['tgt_lang']))

# Adicionar coluna com as traduções
df['translated_by_TA'] = translated_poems

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/mbart/finetuning_musics/frances_portugues.csv", index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
