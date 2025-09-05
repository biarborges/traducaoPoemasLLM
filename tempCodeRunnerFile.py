import pandas as pd

# Função para limpar e padronizar textos
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = text.strip()          # remove espaços no início/fim
    text = text.replace("\n", " ")  # substitui quebras de linha por espaço
    text = " ".join(text.split())   # remove espaços duplicados
    return text

# Caminhos dos arquivos
arquivo_20 = "poemas/test/frances_portugues_test.csv"
arquivo_300 = "poemas/chatgpt/frances_portugues_poems_chatgpt_prompt1.csv"
saida = "poemas/chatgpt/poems_test_prompt1/frances_portugues_20PorCento.csv"

# Carrega os CSVs
df_20 = pd.read_csv(arquivo_20)
df_300 = pd.read_csv(arquivo_300)

# Normaliza textos para merge
df_20["original_norm"] = df_20["original_poem"].apply(normalize_text)
df_300["original_norm"] = df_300["original_poem"].apply(normalize_text)

# Merge mantendo todos do _test
df_merge = pd.merge(
    df_20,
    df_300[["original_norm", "translated_by_TA"]],
    left_on="original_norm",
    right_on="original_norm",
    how="left"
)

# Mantém a ordem original do _test
df_merge = df_merge.set_index("original_norm").reindex(df_20["original_norm"]).reset_index(drop=True)

# Salva apenas as colunas necessárias
df_merge[["original_poem", "translated_poem", "src_lang", "tgt_lang", "translated_by_TA"]].to_csv(saida, index=False, encoding="utf-8")

print(f"Arquivo salvo em: {saida}")
