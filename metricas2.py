import pandas as pd
import nltk
import os
import warnings
warnings.filterwarnings("ignore")

# Verificar se o recurso nltk está disponível
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    print("punkt_tab ok.")
    print("wordnet ok.")
except LookupError:
    print("baixando...")
    nltk.download('punkt_tab')
    nltk.download('wordnet')



input_file = os.path.abspath("poemas/marianmt/frances_ingles_test_finetuning_marianmt.csv")

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

# Função para calcular o BERTScore para um único poema
def calcular_bertscore(referencia, traducao):
    # Calcular o BERTScore entre a referência e a tradução
    P, R, F1 = score([traducao], [referencia], lang="pt")
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

from bartscore import BARTScorer
import pandas as pd

# Função para calcular o BARTScore para um único poema
def calcular_bartscore(referencia, traducao):
    # Inicializar o BARTScorer (usando GPU ou CPU conforme disponível)
    scorer = BARTScorer(device='cuda')  # Ou 'cpu' se não tiver GPU

    # Calcular o BARTScore entre a tradução e a referência
    score = scorer.score([traducao], [referencia])
    return score  # Retorna o BARTScore

# Função para calcular a média do BARTScore de todos os poemas
def calcular_bartscore_media(input_file):
    df = pd.read_csv(input_file)

    # Verificar se as colunas necessárias estão presentes
    if "translated_poem" not in df.columns or "translated_by_TA" not in df.columns:
        raise ValueError("O arquivo CSV deve conter as colunas 'translated_poem' e 'translated_by_TA'.")

    bartscore_scores = []

    # Calcular o BARTScore para cada poema
    for i in range(len(df)):
        referencia_poema = df["translated_poem"].iloc[i]
        traducao_TA = df["translated_by_TA"].iloc[i]

        # Calcular o BARTScore para o poema
        bartscore_value = calcular_bartscore(referencia_poema, traducao_TA)
        bartscore_scores.append(bartscore_value)

    # Calcular a média do BARTScore
    bartscore_media = sum(bartscore_scores) / len(bartscore_scores)

    # Imprimir o resultado
    print(f"Pontuação média BARTScore para todos os poemas: {bartscore_media:.4f}")

# Calcular a média do BARTScore e exibir o resultado
calcular_bartscore_media(input_file)
