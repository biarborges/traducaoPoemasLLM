from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
import pandas as pd
import os
from tqdm import tqdm  
import warnings

warnings.filterwarnings("ignore")

# Caminho do modelo treinado
model_path = os.path.abspath("../modelos/mt5/finetuned_mt5")

# Carregar modelo e tokenizer
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)

# Configurar para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir a língua de origem e destino manualmente
SRC_LANG = "fr"  # Código de linguagem de origem
TGT_LANG = "en"  # Código de linguagem de destino

# Carregar conjunto de teste
test_dataset_path = os.path.abspath("../modelos/poemas/test/frances_ingles_test.csv")
df_test = pd.read_csv(test_dataset_path)  # Carregar o CSV em um DataFrame
print(f"Número de exemplos no conjunto de teste: {len(df_test)}")

# Função para traduzir um poema em partes
def traduzir_poema(texto_origem, chunk_size=256):

    prompt = f"translate {SRC_LANG} to {TGT_LANG}: " + texto_origem
    
    # Tokenizar o texto sem truncar
    tokens = tokenizer(texto_origem, return_tensors="pt", padding=False, truncation=False).input_ids[0].tolist()
    
    # Dividir em partes menores
    partes = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    traducoes = []
    
    for parte in partes:
        input_tensor = torch.tensor([parte], dtype=torch.long).to(device)  # Certificar que é um tensor válido
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=input_tensor,
                max_length=chunk_size * 2,  # Permite mais espaço para tradução
                num_beams=5,  # Opcional: pode usar beam search para melhorar a tradução
                early_stopping=True
            )
        traducoes.append(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
    
    return " ".join(traducoes)

# Traduzir todo o conjunto de teste com barra de progresso
resultados = []

for _, exemplo in tqdm(df_test.iterrows(), desc="Traduzindo poemas", unit="poema", total=len(df_test)):
    poema_original = exemplo["original_poem"]
    referencia = exemplo["translated_poem"]

    # Traduzir o poema
    traducao_gerada = traduzir_poema(poema_original)

    resultados.append({
        "original_poem": poema_original,
        "translated_poem": referencia,
        "translated_by_mt5": traducao_gerada,
        "src_lang": SRC_LANG,
        "tgt_lang": TGT_LANG
    })

# Salvar os resultados em um CSV
df_resultados = pd.DataFrame(resultados)
output_path = os.path.abspath("../modelos/poemas/test/mt5_finetuning/frances_ingles_test_traducao_mt5.csv")
df_resultados.to_csv(output_path, index=False)

print("Traduções salvas.")
