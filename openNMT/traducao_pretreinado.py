import pandas as pd
import subprocess
import os
import time

start_time = time.time()

# Caminhos
CSV_PATH = "../poemas/frances_ingles_poems.csv"
OUTPUT_DIR = "../poemas/openNMT"
INPUT_TXT = os.path.join(OUTPUT_DIR, "input.txt")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "output.txt")
FINAL_CSV = os.path.join(OUTPUT_DIR, "frances_ingles_poems_opennmt.csv")

# Prefixo do idioma de origem (NLLB-200)
SRC_PREFIX = "</s> fra_Latn"  # conforme seu YAML

# Lê o CSV
df = pd.read_csv(CSV_PATH)

# Substitui quebras de linha por marcador temporário e insere prefixo
inputs = df["original_poem"].apply(lambda x: SRC_PREFIX + " " + x.replace('\n', '<br>'))

os.makedirs(OUTPUT_DIR, exist_ok=True)
inputs.to_csv(INPUT_TXT, index=False, header=False)

# Comando do OpenNMT com config.yaml
cmd = [
    "onmt_translate",
    "-config", "../openNMT/config.yaml",  # caminho do seu YAML
    "-src", INPUT_TXT,
    "-output", OUTPUT_TXT,
    "-verbose"
]

# Executa o comando
subprocess.run(cmd)

# Lê o arquivo traduzido e restaura as quebras de linha
with open(OUTPUT_TXT, "r", encoding="utf-8") as f:
    translated_lines = [line.strip().replace("<br>", "\n") for line in f]

# Adiciona ao DataFrame
df["translation_by_TA"] = translated_lines

# Salva CSV final
df.to_csv(FINAL_CSV, index=False)

print(f"Tradução salva em: {FINAL_CSV}")

end_time = time.time()
print(f"Tempo total: {end_time - start_time:.2f} segundos")