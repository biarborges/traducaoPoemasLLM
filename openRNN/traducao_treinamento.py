import pandas as pd
import os
import  time

start_time = time.time()

# Configuração dos modelos de tradução
MODEL_PATHS = {
    ("fr_XX", "pt_XX"): "../openRNN/models_fr_pt/model_fr_pt_step_50000.pt"
    # Adicione mais idiomas se necessário
}

CSV_PATH = "../poemas/poemas300/frances_portugues_poems.csv"
OUTPUT_CSV = "../poemas/poemas300/openRNN/frances_portugues_poems_openRNN.csv"

def traduzir_texto(texto, src, tgt):
    """Traduz um texto usando o modelo adequado"""
    model_path = MODEL_PATHS.get((src, tgt))
    
    if not model_path:
        print(f"⚠️ Modelo não encontrado para {src} → {tgt}")
        return "Erro: Modelo não disponível"

    # Salvar o texto em um arquivo temporário
    with open("temp_input.txt", "w", encoding="utf-8") as f:
        f.write(texto + "\n")

    # Rodar a tradução
    os.system(f"onmt_translate -model {model_path} -src temp_input.txt -output temp_output.txt -gpu 0")

    # Ler o resultado
    with open("temp_output.txt", "r", encoding="utf-8") as f:
        translated_text = f.read().strip()
    
    return translated_text

# 🔹 Carregar os poemas
df = pd.read_csv(CSV_PATH)

# 🔹 Traduzir cada poema
df["translated_by_TA"] = df.apply(
    lambda row: traduzir_texto(row["original_poem"], row["src_lang"], row["tgt_lang"]), axis=1
)

# 🔹 Salvar o novo CSV
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"✅ Tradução concluída! Arquivo salvo como {OUTPUT_CSV}")

end_time = time.time()
print(f"Tempo total: {end_time - start_time:.2f} segundos")
