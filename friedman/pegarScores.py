import pandas as pd
import os

# Dicionário com o nome dos modelos e o caminho dos arquivos
modelos = {
    "openrnn_opus": "ingles_portugues/ingles_portugues_metricas_openRNN.csv",
    "openrnn_ft": "ingles_portugues/ingles_portugues_metricas_openRNN_finetuning.csv",
    "marianmt_opus": "ingles_portugues/ingles_portugues_metricas_marianmt.csv",
    "marianmt_ft": "ingles_portugues/ingles_portugues_metricas_marianmt_finetuning.csv",
    "mbart_opus": "ingles_portugues/ingles_portugues_metricas_mbart.csv",
    "mbart_ft": "ingles_portugues/ingles_portugues_metricas_mbart_finetuning.csv",
    "google": "ingles_portugues/ingles_portugues_metricas_googleTradutor.csv",
    "chatgpt_p1": "ingles_portugues/ingles_portugues_metricas_chatgpt_prompt1.csv",
    "chatgpt_p2": "ingles_portugues/ingles_portugues_metricas_chatgpt_prompt2.csv",
    "maritaca_p1": "ingles_portugues/ingles_portugues_metricas_maritaca_prompt1.csv",
    "maritaca_p2": "ingles_portugues/ingles_portugues_metricas_maritaca_prompt2.csv",
}

# Lista para armazenar os DataFrames com as métricas renomeadas
metricas_por_modelo = []

for nome_modelo, caminho in modelos.items():
    df = pd.read_csv(caminho)

    # Mantém apenas as métricas
    df_metricas = df[["bleu_score", "meteor_score", "bertscore", "bartscore"]].copy()

    # Renomeia as colunas para incluir o nome do modelo
    df_metricas = df_metricas.rename(columns={
        "bleu_score": f"bleu_{nome_modelo}",
        "meteor_score": f"meteor_{nome_modelo}",
        "bertscore": f"bertscore_{nome_modelo}",
        "bartscore": f"bartscore_{nome_modelo}",
    })

    metricas_por_modelo.append(df_metricas)

# Concatenar todas as métricas lado a lado
df_unificado = pd.concat(metricas_por_modelo, axis=1)

# (Opcional) Adicionar ID do poema
df_unificado.insert(0, "poema_id", range(1, len(df_unificado) + 1))

# Salvar CSV unificado
output_path = "metricas_unificadas_ingles_portugues.csv"
df_unificado.to_csv(output_path, index=False)
print(f"✅ CSV unificado salvo em: {output_path}")
