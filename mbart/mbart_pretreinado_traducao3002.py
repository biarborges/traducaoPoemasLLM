import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do mBART
model_name = "/home/ubuntu/finetuning_fr_pt/checkpoint-297"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Função para traduzir um poema linha por linha
def traduzir_poema_em_linhas(poema, src_lang="fr_XX", tgt_lang="pt_XX"):
    if not isinstance(poema, str) or poema.strip() == "":
        return ""

    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    linhas = poema.strip().split('\n')
    traducoes = []

    for linha in linhas:
        if linha.strip() == "":
            traducoes.append("")
            continue

        encoded = tokenizer(linha.strip(), return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_length=512,
                num_beams=5,
                no_repeat_ngram_size=3,
                early_stopping=True,
                forced_bos_token_id=forced_bos_token_id
            )

        traducao = tokenizer.decode(output[0], skip_special_tokens=True)
        traducoes.append(traducao)

    return '\n'.join(traducoes)

# Carregar o arquivo CSV
file_path = "../poemas/test/frances_portugues_test.csv"
df = pd.read_csv(file_path)

# Traduzir os poemas com barra de progresso
translated_poems = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Traduzindo poemas"):
    translated_poems.append(
        traduzir_poema_em_linhas(row['original_poem'], src_lang=row['src_lang'], tgt_lang=row['tgt_lang'])
    )

# Adicionar coluna com as traduções
df['translated_by_TA'] = translated_poems

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/mbart/finetuning_musics/frances_portugues.csv", index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
