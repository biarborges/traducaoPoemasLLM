import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
import torch
import os
import warnings
warnings.filterwarnings("ignore")

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Caminho do modelo treinado
model_path = os.path.abspath("../modelos/mbart/finetuned_mbart")
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)

# Definir idiomas
SRC_LANG = "en_XX"  # Ajuste conforme necessário
TGT_LANG = "pt_XX"  # Ajuste conforme necessário

# Carregar CSV
input_file = os.path.abspath("../modelos/poemas/poemas_unicos/corvo_normalizado.csv")
output_file = os.path.abspath("../modelos/poemas/poemas_unicos/corvo_traduzido_finetuning_mbart.csv")

df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir um poema com barra de progresso por estrofe
def traduzir_texto(texto):
    tokenizer.src_lang = SRC_LANG
    estrofes = texto.split('\n')  # Dividir em estrofes/versos
    
    traducao_completa = []
    for estrofe in tqdm(estrofes, desc="Traduzindo estrofes", leave=False):
        encoded = tokenizer(estrofe, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[TGT_LANG],
            max_length=512,  # Defina explicitamente o max_length
            num_beams=5,  # Número de beams para busca em feixe
            early_stopping=True  # Parar a geração quando atingir um bom resultado
        )
        traducao_completa.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
    
    return '\n'.join(traducao_completa)

# Ativar tqdm no nível do dataframe
tqdm.pandas(desc="Traduzindo poemas...")

# Aplicar a tradução com progresso por poema
df["translated_by_TA"] = df["original_poem"].progress_apply(traduzir_texto)

# Salvar em um novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")
