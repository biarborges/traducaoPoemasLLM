import pandas as pd
import subprocess
import os
import time

start_time = time.time()

# Caminhos
CSV_INPUT = "../poemas/frances_ingles_poems.csv"
CSV_OUTPUT = "../poemas/openNMT/frances_ingles_poems_openNMT.csv"
YAML_CONFIG = "nllb_fr_to_en.yaml"  # ou o nome correto do seu yaml
TEMP_SRC = "temp_src.txt"
TEMP_OUT = "temp_out.txt"

# Carrega o CSV e filtra para francês → inglês
df = pd.read_csv(CSV_INPUT)
df_filtered = df[(df["src_lang"] == "fr") & (df["tgt_lang"] == "en")].copy()

# Salva apenas o texto original (sem prefixo manual)
df_filtered["original_poem"].to_csv(TEMP_SRC, index=False, header=False)

# Roda o OpenNMT com seu arquivo .yaml
subprocess.run([
    "onmt_translate",
    "-config", YAML_CONFIG,
    "-src", TEMP_SRC,
    "-output", TEMP_OUT
])

# Lê a saída do OpenNMT
with open(TEMP_OUT, "r", encoding="utf-8") as f:
    translations = f.read().splitlines()

# Salva as traduções no DataFrame original
df.loc[df_filtered.index, "translated_by_TA"] = translations

# Exporta para o novo CSV
df.to_csv(CSV_OUTPUT, index=False)

# Limpa arquivos temporários
os.remove(TEMP_SRC)
os.remove(TEMP_OUT)

end_time = time.time()
execution_time = end_time - start_time

print("Tradução concluída com sucesso!")
print(f"⏱️ Tempo de execução: {execution_time:.2f} segundos")