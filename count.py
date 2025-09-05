import pandas as pd

# Caminho do arquivo final
saida = "poemas/openRNN/poems_test/portugues_frances_20PorCento.csv"

# Carrega o CSV
df = pd.read_csv(saida)

# Conta quantos poemas tem
total_poemas = len(df)
print(f"Total de poemas: {total_poemas}")

# Verifica se há traduções faltando
faltando = df[df["translated_by_TA"].isna()]

if len(faltando) == 0:
    print("Todos os poemas têm translated_by_TA.")
else:
    print(f"Poemas sem translated_by_TA: {len(faltando)}")
    print("Exemplo de poema sem tradução:")
    print(faltando.iloc[0]["original_poem"])
