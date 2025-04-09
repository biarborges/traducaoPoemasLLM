import pandas as pd
import subprocess
import time
import tempfile
from tqdm import tqdm

# Caminhos
CSV_INPUT = "../poemas/frances_ingles_poems.csv"
CSV_OUTPUT = "../poemas/openNMT/frances_ingles_poems_openNMT.csv"
YAML_CONFIG = "../openNMT/nllb-inference.yaml"

# Carrega o CSV
df = pd.read_csv(CSV_INPUT)

# Filtra as linhas de francês para inglês
df_filtered = df[(df["src_lang"] == "fr") & (df["tgt_lang"] == "en")].copy()

# Substitui quebras de linha para preservar estrutura poética
df_filtered["original_poem"] = df_filtered["original_poem"].apply(
    lambda x: ">>en<< " + str(x).replace("\n", " [LB] ")
)


# Cria arquivos temporários
with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as temp_src, \
     tempfile.NamedTemporaryFile(mode="r", delete=False, encoding="utf-8") as temp_out:

    # Escreve os poemas no arquivo de entrada temporário
    temp_src.write("\n".join(df_filtered["original_poem"].tolist()))
    temp_src.flush()

    print("Iniciando tradução com OpenNMT...")

    # Executa o comando onmt_translate
    subprocess.run([
        "onmt_translate",
        "-config", YAML_CONFIG,
        "-src", temp_src.name,
        "-output", temp_out.name
    ], check=True)

    print("Tradução concluída. Lendo resultados...")

    # Lê as traduções
    temp_out.seek(0)
    translations = [line.strip().replace(' [LB] ', '\n') for line in temp_out.readlines()]

# Atribui as traduções ao DataFrame original
for idx, trans in tqdm(zip(df_filtered.index, translations), total=len(translations), desc="Salvando traduções"):
    df.loc[idx, "translated_by_TA"] = trans

# Salva o novo CSV
df.to_csv(CSV_OUTPUT, index=False)

print("✅ Tradução concluída e salva com sucesso.")
