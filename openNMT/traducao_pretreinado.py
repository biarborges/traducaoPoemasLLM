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

# Carrega o CSV e filtra francês → inglês
df = pd.read_csv(CSV_INPUT)
df_filtered = df[(df["src_lang"] == "fr") & (df["tgt_lang"] == "en")].copy()

# Remove vazios
df_filtered = df_filtered[df_filtered["original_poem"].notnull()]
df_filtered = df_filtered[df_filtered["original_poem"].str.strip() != ""]

# Substitui quebras por marcador temporário e adiciona token do idioma
df_filtered["poem_for_translation"] = df_filtered["original_poem"].apply(
    lambda x: f">>en<< {str(x).replace('\n', ' [LINEBREAK] ')}")

# Cria arquivos temporários
with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_src, \
     tempfile.NamedTemporaryFile(mode="r", delete=False) as temp_out:

    # Escreve no arquivo de entrada
    df_filtered["poem_for_translation"].to_csv(temp_src.name, index=False, header=False)

    # Verificação de conteúdo do arquivo temporário
    with open(temp_src.name, "r", encoding="utf-8") as debug_f:
        lines = debug_f.readlines()
        print(f"⚠️ Total de linhas no arquivo de entrada: {len(lines)}")
        if lines:
            print("🔍 Exemplo da 1ª linha:", lines[0][:200])

    print("\n🚀 Iniciando tradução com OpenNMT...")

    # Tradução
    subprocess.run([
        "onmt_translate",
        "-config", YAML_CONFIG,
        "-src", temp_src.name,
        "-output", temp_out.name
    ])

    print("✅ Tradução concluída! Processando resultados...")

    # Lê traduções
    with open(temp_out.name, "r", encoding="utf-8") as f:
        translations = [line.replace(" [LINEBREAK] ", "\n") for line in f.read().splitlines()]

# Salva no DataFrame original
for idx, trans in tqdm(zip(df_filtered.index, translations), total=len(translations), desc="Salvando traduções"):
    df.loc[idx, "translated_by_TA"] = trans

# Exporta CSV final
df.to_csv(CSV_OUTPUT, index=False)

end_time = time.time()
print("✅ Tradução salva com sucesso!")
print(f"⏱️ Tempo total de execução: {end_time - start_time:.2f} segundos")
