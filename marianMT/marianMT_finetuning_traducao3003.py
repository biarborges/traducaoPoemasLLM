import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do MarianMT
model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Caminhos dos arquivos
input_file = "../poemas/poemas300/test/frances_ingles_test.csv"
output_file = "../poemas/poemas300/marianmt/frances_ingles_test_pretreinado_marianmt.csv"

# Carregar CSV
df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir um poema
def traduzir_texto(texto):
    texto_com_prefixo = f">>en<< {texto}"
    estrofes = texto_com_prefixo.split('\n')
    traducao_completa = []
    
    for estrofe in estrofes:
        encoded = tokenizer(estrofe, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            generated_tokens = model.generate(**encoded)
        traducao_completa.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
    
    return '\n'.join(traducao_completa)

# Aplicar a tradução para cada poema com barra de progresso
tqdm.pandas(desc="Traduzindo poemas")
df["translated_by_marian"] = df["original_poem"].progress_apply(traduzir_texto)

# Reorganizar as colunas
df = df[["original_poem", "translated_poem", "translated_by_marian", "src_lang", "tgt_lang"]]

# Salvar em um novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")
