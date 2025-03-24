#igual ao 300, só q pra portugues como lingua source

import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do mBART
model_name = "/home/ubuntu/finetuning_fr_ing/checkpoint-45"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Função para traduzir poema
def traduzir_poema(poema, src_lang="fr_XX", tgt_lang="en_XX"):
    if not isinstance(poema, str) or poema.strip() == "":
        return ""  # Evitar erros com valores nulos ou vazios

    # Configurar o idioma correto no tokenizer
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    # Tokenizar o texto com a configuração correta
    encoded = tokenizer(poema.strip(), return_tensors="pt", truncation=True, padding=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU

    # Forçar o idioma de saída correto
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]

    # Gerar tradução
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            max_length=512,
            num_beams=5,
            forced_bos_token_id=forced_bos_token_id  # Isso garante que a saída seja no idioma correto
        )

    traducao = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return traducao

# Carregar o arquivo CSV com os poemas
file_path = "../poemas/poemas300/test/frances_ingles_test.csv"
df = pd.read_csv(file_path)

# Exibir para verificar o conteúdo e garantir que 'src_lang' e 'tgt_lang' estejam corretos
print(df[['original_poem', 'src_lang', 'tgt_lang']].head())

# Traduzir os poemas e adicionar à nova coluna 'translated_by_TA'
df['translated_by_TA'] = df.apply(lambda row: traduzir_poema(row['original_poem'], src_lang=row['src_lang'], tgt_lang=row['tgt_lang']), axis=1)

# Salvar o resultado em um novo CSV
df.to_csv("../poemas/poemas300/mbart/frances_ingles_test_finetuning_mbart.csv", index=False)

print("Tradução concluída e salva.")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
