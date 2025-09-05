import pandas as pd

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text

arquivo_20 = "poemas/test/portugues_frances_test.csv"
arquivo_300 = "poemas/openRNN/portugues_frances_poems_openRNN.csv"
saida = "poemas/openRNN/poems_test/portugues_frances_20PorCento.csv"

# Carrega os CSVs
df_20 = pd.read_csv(arquivo_20)
df_300 = pd.read_csv(arquivo_300)

# Normaliza textos para merge
df_20["original_norm"] = df_20["original_poem"].apply(normalize_text)
df_20["translated_norm"] = df_20["translated_poem"].apply(normalize_text)

df_300["original_norm"] = df_300["original_poem"].apply(normalize_text)
df_300["translated_norm"] = df_300["translated_poem"].apply(normalize_text)

# Merge usando as duas colunas como chave
df_merge = pd.merge(
    df_20,
    df_300[["original_norm", "translated_norm", "translated_by_TA"]],
    left_on=["original_norm", "translated_norm"],
    right_on=["original_norm", "translated_norm"],
    how="left",
    sort=False
)

# Salva apenas as colunas finais
df_merge[["original_poem", "translated_poem", "src_lang", "tgt_lang", "translated_by_TA"]].to_csv(saida, index=False, encoding="utf-8")

print(f"Arquivo salvo em: {saida}")
