import pandas as pd
import subprocess
import time
import tempfile
from tqdm import tqdm

start_time = time.time()

# Caminhos
CSV_INPUT = "../poemas/frances_ingles_poems.csv"
CSV_OUTPUT = "../poemas/openNMT/frances_ingles_poems_openNMT.csv"
YAML_CONFIG = "../openNMT/nllb-inference.yaml"

# Carrega o CSV e filtra para francês → inglês
df = pd.read_csv(CSV_INPUT)
df_filtered = df[(df["src_lang"] == "fr") & (df["tgt_lang"] == "en")].copy()

# Cria arquivos temporários
with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_src, \
     tempfile.NamedTemporaryFile(mode="r", delete=False) as temp_out:

    # Escreve os poemas no arquivo de entrada temporário
    df_filtered["original_poem"].to_csv(temp_src.name, index=False, header=False)

    print("Iniciando tradução com OpenNMT...")

    # Chama o OpenNMT
    subprocess.run([
        "onmt_translate",
        "-config", YAML_CONFIG,
        "-src", temp_src.name,
        "-output", temp_out.name
    ])

    print("Tradução concluída! Processando resultados...")

    # Lê as traduções
    with open(temp_out.name, "r", encoding="utf-8") as f:
        translations = f.read().splitlines()

# Atualiza o DataFrame
for idx, trans in tqdm(zip(df_filtered.index, translations), total=len(translations), desc="Salvando traduções"):
    df.loc[idx, "translated_by_TA"] = trans

# Exporta o CSV
df.to_csv(CSV_OUTPUT, index=False)

end_time = time.time()
print("✅ Tradução concluída com sucesso!")
print(f"⏱️ Tempo de execução: {end_time - start_time:.2f} segundos")
