import pandas as pd
import subprocess
import os
import sentencepiece as spm
from tqdm import tqdm

# Caminhos
CSV_PATH = "../poemas/frances_ingles_poems_teste.csv"
OUT_DIR = "../poemas/OpenNMT"
MODEL_PATH = "nllb-200-1.3B-onmt.pt"
SP_MODEL_PATH = "flores200_sacrebleu_tokenizer_spm.model"
SRC_LANG_TAG = ">>eng_Latn<<"  # Linguagem de destino
TMP_SRC = "tmp_input.txt"
TMP_TGT = "tmp_output.txt"

# Inicializa SentencePiece
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# Carrega o CSV
df = pd.read_csv(CSV_PATH)

# Substitui quebras de linha por marcador temporário e aplica tokenização
def preprocess_poem(poem):
    poem = poem.replace("\n", " <n> ").strip()
    tagged = f"{SRC_LANG_TAG} {poem}"
    tokenized = sp.encode(tagged, out_type=str)
    return " ".join(tokenized)

print("Pré-processando e tokenizando os poemas...")
preprocessed_poems = df["original_poem"].apply(preprocess_poem)

# Salva os poemas tokenizados no arquivo temporário
with open(TMP_SRC, "w", encoding="utf-8") as f:
    for line in preprocessed_poems:
        f.write(line + "\n")

# Comando de tradução (sem tokenizer/spm no argumento)
command = [
    "python3", "-m", "onmt.bin.translate",
    "-model", MODEL_PATH,
    "-src", TMP_SRC,
    "-output", TMP_TGT,
    "-gpu", "0"  # Altere para "-gpu", "-1" se quiser usar CPU
]

print("Traduzindo com OpenNMT...")
subprocess.run(command, check=True)
print("Tradução concluída.")

# Lê a saída traduzida, aplica detokenização e restaura quebras de linha
with open(TMP_TGT, "r", encoding="utf-8") as f:
    translated_lines = [sp.decode(line.strip().split()).replace(" <n> ", "\n") for line in f]

# Adiciona ao DataFrame e salva
df["translated_by_TA"] = translated_lines

# Cria diretório de saída se necessário
os.makedirs(OUT_DIR, exist_ok=True)

# Salva CSV
OUT_PATH = os.path.join(OUT_DIR, os.path.basename(CSV_PATH))
df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"Arquivo salvo em: {OUT_PATH}")
