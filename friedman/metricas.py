import pandas as pd
import nltk
import os
import warnings
import torch
import time
warnings.filterwarnings("ignore")

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


#Verificar se o recurso nltk está disponível
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    print("punkt_tab ok.")
    print("wordnet ok.")
except LookupError:
    print("baixando...")
    nltk.download('punkt_tab')
    nltk.download('wordnet')

input_file = os.path.abspath("poemas/chatgpt/frances_ingles_poems_chatgpt_prompt1.csv")
output_csv = input_file.replace(".csv", "friedman/frances_ingles_chatGPTPrompt1.csv")
lang = "en"  # tradução

print(f"Arquivo de entrada: {input_file}")

#BLEU

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Função para calcular o BLEU para um único poema
def calcular_bleu(referencia, traducao):
    # Tokenizar as referências e a tradução
    referencia_tokens = [nltk.word_tokenize(referencia.lower())]
    traducao_tokens = nltk.word_tokenize(traducao.lower())

    # Calcular BLEU com suavização para evitar pontuações zero
    smooth = SmoothingFunction().method4
    return sentence_bleu(referencia_tokens, traducao_tokens, smoothing_function=smooth)

# Função para calcular a média do BLEU de todos os poemas
def calcular_bleu_media(input_file):
    df = pd.read_csv(input_file)

    # Verificar se as colunas necessárias estão presentes
    if "translated_poem" not in df.columns or "translated_by_TA" not in df.columns:
        raise ValueError("O arquivo CSV deve conter as colunas 'translated_poem' e 'translated_by_TA'.")

    bleu_scores = []

    # Calcular o BLEU para cada poema
    for i in range(len(df)):
        referencia_poema = df["translated_poem"].iloc[i]
        traducao_TA = df["translated_by_TA"].iloc[i]

        # Calcular o BLEU para o poema
        bleu_score = calcular_bleu(referencia_poema, traducao_TA)
        bleu_scores.append(bleu_score)

    # Calcular a média do BLEU
    bleu_media = sum(bleu_scores) / len(bleu_scores)

    # Imprimir o resultado
    print(f"Pontuação média BLEU para todos os poemas: {bleu_media:.4f}")

# Calcular a média do BLEU e exibir o resultado
calcular_bleu_media(input_file)

#METEOR

from nltk.translate.meteor_score import meteor_score

# Função para calcular a METEOR para um único poema
def calcular_meteor(referencia, traducao):
    # Tokenizar as referências e a tradução
    referencia_tokens = nltk.word_tokenize(referencia.lower())
    traducao_tokens = nltk.word_tokenize(traducao.lower())

    # Calcular METEOR
    return meteor_score([referencia_tokens], traducao_tokens)

# Função para calcular a média da METEOR de todos os poemas
def calcular_meteor_media(input_file):
    df = pd.read_csv(input_file)

    # Verificar se as colunas necessárias estão presentes
    if "translated_poem" not in df.columns or "translated_by_TA" not in df.columns:
        raise ValueError("O arquivo CSV deve conter as colunas 'translated_poem' e 'translated_by_TA'.")

    meteor_scores = []

    # Calcular a METEOR para cada poema
    for i in range(len(df)):
        referencia_poema = df["translated_poem"].iloc[i]
        traducao_TA = df["translated_by_TA"].iloc[i]

        # Calcular a METEOR para o poema
        meteor_score_value = calcular_meteor(referencia_poema, traducao_TA)
        meteor_scores.append(meteor_score_value)

    # Calcular a média da METEOR
    meteor_media = sum(meteor_scores) / len(meteor_scores)

    # Imprimir o resultado
    print(f"Pontuação média METEOR para todos os poemas: {meteor_media:.4f}")

# Calcular a média da METEOR e exibir o resultado
calcular_meteor_media(input_file)


#BERTSCORE

from bert_score import score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Função para calcular o BERTScore para um único poema
def calcular_bertscore(referencia, traducao):
    # Calcular o BERTScore entre a referência e a tradução
    P, R, F1 = score([traducao], [referencia], lang=lang, device=device)
    return F1.mean().item()  # Retornar a pontuação F1 média (similaridade semântica)

# Função para calcular a média do BERTScore de todos os poemas
def calcular_bertscore_media(input_file):
    df = pd.read_csv(input_file)

    # Verificar se as colunas necessárias estão presentes
    if "translated_poem" not in df.columns or "translated_by_TA" not in df.columns:
        raise ValueError("O arquivo CSV deve conter as colunas 'translated_poem' e 'translated_by_TA'.")

    bertscore_scores = []

    # Calcular o BERTScore para cada poema
    for i in range(len(df)):
        referencia_poema = df["translated_poem"].iloc[i]
        traducao_TA = df["translated_by_TA"].iloc[i]

        # Calcular o BERTScore para o poema
        bertscore_value = calcular_bertscore(referencia_poema, traducao_TA)
        bertscore_scores.append(bertscore_value)

    # Calcular a média do BERTScore
    bertscore_media = sum(bertscore_scores) / len(bertscore_scores)

    # Imprimir o resultado
    print(f"Pontuação média BERTScore para todos os poemas: {bertscore_media:.4f}")

# Calcular a média do BERTScore e exibir o resultado
calcular_bertscore_media(input_file)


#BARTSCORE

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Carregar o modelo BART e o tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)  


# Função para calcular o BARTScore
def calcular_bartscore(referencia, traducao):
    # Tokenizar a referência e a tradução
    inputs_ref = tokenizer(referencia, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    inputs_trad = tokenizer(traducao, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)


    # Gerar os embeddings de BART para a referência e tradução
    with torch.no_grad():
        embeddings_ref = model.get_encoder()(inputs_ref['input_ids'])[0]
        embeddings_trad = model.get_encoder()(inputs_trad['input_ids'])[0]

    # Calcular similaridade de cosseno entre os embeddings
    cosine_sim = cosine_similarity(embeddings_ref.mean(dim=1).cpu().numpy(), embeddings_trad.mean(dim=1).cpu().numpy())

    return cosine_sim[0][0]

# Função para calcular o BARTScore para todos os poemas de forma paralela
def calcular_bartscore_media_paralela(input_file):
    df = pd.read_csv(input_file)

    # Verificar se as colunas necessárias estão presentes
    if "translated_poem" not in df.columns or "translated_by_TA" not in df.columns:
        raise ValueError("O arquivo CSV deve conter as colunas 'translated_poem' e 'translated_by_TA'.")

    # Usar ThreadPoolExecutor para processar os poemas em paralelo
    with ThreadPoolExecutor() as executor:
        # Calcular o BARTScore para todos os poemas em paralelo
        bartscore_scores = list(executor.map(calcular_bartscore,
                                            df["translated_poem"],
                                            df["translated_by_TA"]))

    # Calcular a média do BARTScore
    bartscore_media = sum(bartscore_scores) / len(bartscore_scores)

    # Imprimir o resultado
    print(f"Pontuação média BARTScore para todos os poemas: {bartscore_media:.4f}")

# Calcular o BARTScore médio para todos os poemas e exibir o resultado
calcular_bartscore_media_paralela(input_file)


end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")


# === Consolidar resultados por poema ===

# Recarregar o DataFrame para manter as colunas originais
df = pd.read_csv(input_file)

# Recalcular todas as métricas por poema e armazenar os valores individuais
bleu_scores = []
meteor_scores = []
bertscore_scores = []
bartscore_scores = []

print("📊 Calculando todas as métricas por poema...")

for i in tqdm(range(len(df))):
    ref = df["translated_poem"].iloc[i]
    hyp = df["translated_by_TA"].iloc[i]

    bleu = calcular_bleu(ref, hyp)
    meteor = calcular_meteor(ref, hyp)
    bert = calcular_bertscore(ref, hyp)
    bart = calcular_bartscore(ref, hyp)

    bleu_scores.append(bleu)
    meteor_scores.append(meteor)
    bertscore_scores.append(bert)
    bartscore_scores.append(bart)

# Adiciona as colunas com os resultados ao DataFrame
df["bleu_score"] = bleu_scores
df["meteor_score"] = meteor_scores
df["bertscore"] = bertscore_scores
df["bartscore"] = bartscore_scores

# Salvar o CSV com as métricas
df.to_csv(output_csv, index=False)
print(f"✅ Resultados salvos em: {output_csv}")
