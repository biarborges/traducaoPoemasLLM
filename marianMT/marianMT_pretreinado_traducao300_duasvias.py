import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import os
import torch
from tqdm import tqdm

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Modelos para tradução
model_1 = "Helsinki-NLP/opus-mt-fr-en"  # Primeira etapa 
model_2 = "Helsinki-NLP/opus-mt-tc-big-en-pt"  # Segunda etapa 

# Carregar tokenizers e modelos
tokenizer_1 = MarianTokenizer.from_pretrained(model_1)
model_1 = MarianMTModel.from_pretrained(model_1).to(device)

tokenizer_2 = MarianTokenizer.from_pretrained(model_2)
model_2 = MarianMTModel.from_pretrained(model_2).to(device)

# Carregar CSV
input_file = os.path.abspath("../poemas/poemas300/test/frances_portugues_test.csv")
output_file = os.path.abspath("../poemas/poemas300/marianmt/frances_portugues_test_pretreinado_marianmt.csv")

df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir um texto com um modelo específico
def traduzir_texto(texto, tokenizer, model, src_lang, tgt_lang):
    try:
        texto_com_prefixo = f">>{tgt_lang}<< {texto}"  # Adiciona o prefixo da língua alvo
        estrofes = texto_com_prefixo.split('\n')
        traducao_completa = []

        for estrofe in tqdm(estrofes, desc=f"Traduzindo {src_lang} → {tgt_lang}", unit="estrofe"):
            if estrofe.strip():  # Evitar linhas vazias
                encoded = tokenizer(estrofe, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
                encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU
                with torch.no_grad():
                    generated_tokens = model.generate(**encoded)
                traducao_completa.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))

        return '\n'.join(traducao_completa)
    except Exception as e:
        print(f"Erro ao traduzir: {texto[:30]}... Erro: {e}")
        return ""

# Função para traduzir primeiro para EN e depois para PT
def traduzir_duas_etapas(row):
    print(f"Traduzindo poema {row.name + 1}/{len(df)}")

    # Primeira etapa: Francês → Inglês
    traducao_en = traduzir_texto(row["original_poem"], tokenizer_1, model_1, "fr", "en")

    # Segunda etapa: Inglês → Português
    traducao_pt = traduzir_texto(traducao_en, tokenizer_2, model_2, "en", "pt")

    return traducao_pt

# Aplicar a tradução com barra de progresso
tqdm.pandas(desc="Traduzindo poemas")
df["translated_by_marian"] = df.progress_apply(traduzir_duas_etapas, axis=1)

# Reorganizar colunas
df["src_lang"] = "fr"
df["tgt_lang"] = "pt"
df = df[["original_poem", "translated_poem", "translated_by_marian", "src_lang", "tgt_lang"]]

# Salvar em CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")
