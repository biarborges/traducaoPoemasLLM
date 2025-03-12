import pandas as pd
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from tqdm import tqdm
import torch
import os
import warnings
warnings.filterwarnings("ignore")

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Caminho do modelo fine-tunado
model_path = os.path.abspath("../modelos/mt5/finetuned_mt5")
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)

# Definir idiomas
SRC_LANG = "en"  # Idioma de origem (inglês)
TGT_LANG = "pt"  # Idioma de destino (português)

# Carregar CSV
input_file = os.path.abspath("../modelos/poemas/poemas_unicos/corvo_normalizado.csv")
output_file = os.path.abspath("../modelos/poemas/poemas_unicos/corvo_traduzido_finetuning_mt5.csv")

df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir um poema com barra de progresso por estrofe
def traduzir_texto(texto):
    try:
        # Adicionar o prefixo de tradução ao texto
        texto_com_prefixo = f"translate {SRC_LANG} to {TGT_LANG}: {texto}"
        
        # Tokenizar o texto
        encoded = tokenizer(texto_com_prefixo, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        
        # Gerar a tradução
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                max_length=512,  # Defina explicitamente o max_length
                num_beams=5,  # Número de beams para busca em feixe
                early_stopping=True,  # Parar a geração quando atingir um bom resultado
                no_repeat_ngram_size=2  # Evitar repetições de n-gramas
            )
        
        # Decodificar a tradução
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Erro ao traduzir o texto: {texto}. Erro: {e}")
        return ""  # Retorna uma string vazia em caso de erro

# Ativar tqdm no nível do dataframe
tqdm.pandas(desc="Traduzindo poemas...")

# Aplicar a tradução com progresso por poema
df["translated_by_mt5"] = df["original_poem"].progress_apply(traduzir_texto)

# Verificar se todas as traduções estão vazias
if df["translated_by_mt5"].empty or df["translated_by_mt5"].isna().all():
    print("Aviso: Nenhuma tradução foi gerada. Verifique o texto de entrada e o modelo.")

# Salvar em um novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")