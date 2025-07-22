from scipy.stats import friedmanchisquare
import pandas as pd

df = pd.read_csv("metricas_unificadas_portugues_ingles.csv")

# 🔍 Pegar apenas colunas que NÃO contêm "_ft"
def get_cols_sem_ft(metrica):
    return [col for col in df.columns if col.startswith(metrica) and "_ft" not in col]

metricas = ["bleu", "meteor", "bertscore", "bartscore"]

print("==== 🔹 COMPARAÇÃO ENTRE MODELOS COMPLETOS (300 poemas) ====")

for metrica in metricas:
    cols = get_cols_sem_ft(metrica)
    df_metrica = df[cols].dropna()

    dados = [df_metrica[col].values for col in cols]

    estatistica, p_valor = friedmanchisquare(*dados)
    print(f"\n📊 {metrica.upper()} ({len(df_metrica)} poemas):")
    print(f"Estatística = {estatistica:.4f} | p-valor = {p_valor:.10e}")
    print("➡️", "Diferença significativa!" if p_valor < 0.05 else "Sem diferença significativa.")



# 🔍 Pegar colunas de todas as métricas
def get_cols_todos(metrica):
    return [col for col in df.columns if col.startswith(metrica)]

print("\n\n==== 🔹 COMPARAÇÃO ENTRE TODOS OS MODELOS (30 poemas) ====")

for metrica in metricas:
    cols = get_cols_todos(metrica)
    
    # Pegar apenas as linhas que têm todos os valores da métrica (ex: 30 poemas com FT)
    df_metrica = df[cols].dropna()
    dados = [df_metrica[col].values for col in cols]

    if len(df_metrica) < 2:
        print(f"\n⚠️ {metrica.upper()}: Menos de 2 amostras disponíveis.")
        continue

    estatistica, p_valor = friedmanchisquare(*dados)
    print(f"\n📊 {metrica.upper()} ({len(df_metrica)} poemas):")
    print(f"Estatística = {estatistica:.4f} | p-valor = {p_valor:.10e}")
    print("➡️", "Diferença significativa!" if p_valor < 0.05 else "Sem diferença significativa.")
