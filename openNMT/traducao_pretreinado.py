import pandas as pd
import subprocess
import os
from tqdm import tqdm

# Caminhos
CSV_PATH = "../traducaoPoemasLLM/poemas/ingles_frances_poems_teste.csv"
OUT_DIR = "../traducaoPoemasLLM/poemas/OpenNMT/frances_ingles_poems_OpenNMT.csv"
MODEL_PATH = "nllb-200-1.3B-onmt.pt"
SP_MODEL_PATH = "flores200_sacrebleu_tokenizer_spm.model"
SRC_LANG_TAG = ">>eng_Latn<<"
TMP_SRC = "tmp_input.txt"
TMP_TGT = "tmp_output.txt"

# Carrega o CSV
df = pd.read_csv(CSV_PATH)

# Substitui quebras de linha por marcador temporário
def preprocess_poem(poem):
    poem = poem.replace("\n", " <n> ")
    return f"{SRC_LANG_TAG} {poem.strip()}"

# Processamento
preprocessed_poems = df["original_poem"].apply(preprocess_poem)

# Salva os poemas preprocessados
with open(TMP_SRC, "w", encoding="utf-8") as f:
    for line in preprocessed_poems:
        f.write(line + "\n")

# Comando de tradução
command = [
    "onmt_translate",
    "-model", MODEL_PATH,
    "-src", TMP_SRC,
    "-output", TMP_TGT,
    "-tokenizer", "spm",
    "-spm_model_path", SP_MODEL_PATH,
    "-gpu", "0"  # Remova ou altere se não estiver usando GPU
]

print("Traduzindo com OpenNMT...")
subprocess.run(command, check=True)
print("Tradução concluída.")

# Lê os resultados e restaura as quebras de linha
with open(TMP_TGT, "r", encoding="utf-8") as f:
    translated_lines = [line.strip().replace(" <n> ", "\n") for line in f]

# Adiciona ao DataFrame e salva
df["translated_by_TA"] = translated_lines

os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, os.path.basename(CSV_PATH))
df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"Arquivo salvo em: {OUT_PATH}")
