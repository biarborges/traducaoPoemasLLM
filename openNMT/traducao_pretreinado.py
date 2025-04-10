import pandas as pd
import os
import subprocess
import time
import torch

start_time = time.time()

# Caminhos
CSV_PATH = "../poemas/frances_portugues_poems.csv"
OUTPUT_CSV = "../poemas/openNMT/frances_portugues_poems_opennmt_old.csv"
TEMP_INPUT = "../poemas/openNMT/temp_input.txt"
TEMP_OUTPUT = "../poemas/openNMT/output.txt"
CONFIG_PATH = "../openNMT/config.yaml"
BREAK_TOKEN = "<br>"

# Carrega CSV
df = pd.read_csv(CSV_PATH)
poemas = df["original_poem"].tolist()

# Tradução por lotes
batch_size = 100
translated_poemas = []

for i in range(0, len(poemas), batch_size):
    batch = poemas[i:i+batch_size]
    print(f"Traduzindo blocos {i} até {i + len(batch) - 1}")

    # Substitui as quebras de linha por um marcador
    batch_processed = [p.replace("\n", BREAK_TOKEN) for p in batch]

    # Adicionando o prefixo da língua de origem diretamente no texto
    processed_batch = [f">>eng_Latn<< {p}" for p in batch_processed]

    # Salva entrada
    with open(TEMP_INPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_batch))

    # Chama o OpenNMT
    result = subprocess.run([
        "onmt_translate",
        "-config", CONFIG_PATH,
        "-src", TEMP_INPUT,
        "-output", TEMP_OUTPUT,
        "-verbose"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Erro ao executar onmt_translate:")
        print(result.stderr)
        exit(1)

    # Lê a saída e restaura quebras de linha
    with open(TEMP_OUTPUT, "r", encoding="utf-8") as f:
        translated = [line.strip().replace(BREAK_TOKEN, "\n") for line in f.readlines()]
        translated_poemas.extend(translated)

    # Liberar memória da GPU entre as traduções
    torch.cuda.empty_cache()

# Valida quantidade de traduções
if len(translated_poemas) != len(poemas):
    raise ValueError(f"⚠️ Número de traduções ({len(translated_poemas)}) difere do número de poemas ({len(poemas)})")

# Salva no CSV
df["translation_by_TA"] = translated_poemas
df.to_csv(OUTPUT_CSV, index=False)

print("Tradução concluída e salva em:", OUTPUT_CSV)

end_time = time.time()
print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")
