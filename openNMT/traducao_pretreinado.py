import pandas as pd
import os
import subprocess

# Caminhos e constantes
CSV_PATH = "../poemas/ingles_frances_poems.csv"
OUTPUT_CSV = "../poemas/openNMT/ingles_frances_poems_traduzido.csv"
TEMP_INPUT = "../poemas/openNMT/temp_input.txt"
TEMP_OUTPUT = "../poemas/openNMT/output.txt"
CONFIG_PATH = "../openNMT/config.yaml"
SRC_PREFIX = "</s> fra_Latn"
BREAK_TOKEN = "<br>"

# Carregar CSV
df = pd.read_csv(CSV_PATH)
poemas = df["original_poem"].tolist()

# Dividir em blocos
batch_size = 50
translated_poemas = []

for i in range(0, len(poemas), batch_size):
    batch = poemas[i:i+batch_size]
    print(f"Traduzindo blocos {i} até {i + len(batch) - 1}")

    # Substituir \n por <br> e adicionar prefixo
    processed_batch = [SRC_PREFIX + " " + p.replace("\n", BREAK_TOKEN) for p in batch]

    # Salvar arquivo temporário para entrada
    with open(TEMP_INPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_batch))

    # Chamar OpenNMT
    subprocess.run([
        "onmt_translate",
        "-config", CONFIG_PATH,
        "-src", TEMP_INPUT,
        "-output", TEMP_OUTPUT,
        "-verbose"
    ])

    # Ler a saída e restaurar quebras de linha
    with open(TEMP_OUTPUT, "r", encoding="utf-8") as f:
        translated = [line.strip().replace(BREAK_TOKEN, "\n") for line in f.readlines()]
        translated_poemas.extend(translated)

# Verificar se número de traduções está correto
if len(translated_poemas) != len(poemas):
    raise ValueError(f"Número de traduções ({len(translated_poemas)}) difere do número de poemas ({len(poemas)})")

# Salvar no CSV
df["translation_by_TA"] = translated_poemas
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Tradução concluída e salva em:", OUTPUT_CSV)
