import pandas as pd
import nltk
import os
import warnings
import torch
import time
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Verificar se o recurso nltk está disponível
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

input_file = os.path.abspath("../poemas/chatgpt/frances_ingles_poems_chatgpt_prompt1.csv")
output_csv = input_file.replace(".csv", "frances_ingles_metricas_chatgpt_prompt1.csv")
lang = "en"  # idioma da tradução

print(f"Arquivo de entrada: {input_file}")

# Função para calcular BLEU
def calcular_bleu(referencia, traducao):
    referencia_tokens = [nltk.word_tokenize(referencia.lower())]
    traducao_tokens = nltk.word_tokenize(traducao.lower())
    smooth = SmoothingFunction().method4
    return sentence_bleu(referencia_tokens, traducao_tokens, smoothing_function=smooth)

# Função para calcular METEOR
def calcular_meteor(referencia, traducao):
    referencia_tokens = nltk.word_tokenize(referencia.lower())
    traducao_tokens = nltk.word_tokenize(traducao.lower())
    return meteor_score([referencia_tokens], traducao_tokens)

# Função para calcular BERTScore
def calcular_bertscore(referencia, traducao):
    P, R, F1 = score([traducao], [referencia], lang=lang, device=device, verbose=False)
    return F1.mean().item()

# Inicializar modelo BART e tokenizer para BARTScore
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

# Função para calcular BARTScore
def calcular_bartscore(referencia, traducao):
    inputs_ref = tokenizer(referencia, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    inputs_trad = tokenizer(traducao, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)

    with torch.no_grad():
        embeddings_ref = model.get_encoder()(inputs_ref['input_ids'])[0]
        embeddings_trad = model.get_encoder()(inputs_trad['input_ids'])[0]

    cosine_sim = cosine_similarity(embeddings_ref.mean(dim=1).cpu().numpy(), embeddings_trad.mean(dim=1).cpu().numpy())
    return cosine_sim[0][0]

# Carregar dados
df = pd.read_csv(input_file)

# Listas para armazenar os scores
bleu_scores = []
meteor_scores = []
bertscore_scores = []
bartscore_scores = []

print("📊 Calculando métricas individuais para cada poema...")

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

# Adicionar colunas ao DataFrame
df["bleu_score"] = bleu_scores
df["meteor_score"] = meteor_scores
df["bertscore"] = bertscore_scores
df["bartscore"] = bartscore_scores

# Salvar CSV
df.to_csv(output_csv, index=False)
print(f"✅ Resultados individuais salvos em: {output_csv}")

end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")
