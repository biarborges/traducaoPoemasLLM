import torch
import os
import pandas as pd
from bert_score import score
import transformers
from tqdm import tqdm  # Barra de progresso

# Define o nível de "verbosidade" do transformers para apenas mostrar erros.
transformers.utils.logging.set_verbosity_error()

input_file = os.path.abspath("../modelagemTopicos/results/frances_ingles_original/chatGPTPrompt1/topico_0.csv")
lang = "en"

print(f"Arquivo de entrada: {input_file}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Função para calcular o BERTScore para um único poema
def calcular_bertscore(referencia, traducao):
    P, R, F1 = score([traducao], [referencia], lang=lang, device=device)
    return F1.mean().item()

# Função para calcular a média do BERTScore de todos os poemas
def calcular_bertscore_media(input_file):
    df = pd.read_csv(input_file)

    if "translated_poem" not in df.columns or "translated_by_TA" not in df.columns:
        raise ValueError("O arquivo CSV deve conter as colunas 'translated_poem' e 'translated_by_TA'.")

    bertscore_scores = []

    # Usando tqdm para a barra de progresso
    for i in tqdm(range(len(df)), desc="Calculando BERTScore"):
        referencia_poema = df["translated_poem"].iloc[i]
        traducao_TA = df["translated_by_TA"].iloc[i]

        bertscore_value = calcular_bertscore(referencia_poema, traducao_TA)
        bertscore_scores.append(bertscore_value)

    bertscore_media = sum(bertscore_scores) / len(bertscore_scores)

    print(f"\n✅ Pontuação média BERTScore para todos os poemas: {bertscore_media:.4f}")

# Executa
calcular_bertscore_media(input_file)
