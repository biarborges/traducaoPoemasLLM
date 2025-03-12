import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch
import os
import warnings

warnings.filterwarnings("ignore")

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar o modelo multilíngue
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"  # Modelo para tradução de inglês para línguas românicas
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Carregar CSV
input_file = os.path.abspath("../modelos/poemas/poemas_unicos/corvo_normalizado.csv")
output_file = os.path.abspath("../modelos/poemas/poemas_unicos/corvo_traduzido_pretreinado_marianmt.csv")

df = pd.read_csv(input_file)

# Verifica se a coluna original_poem existe
if "original_poem" not in df.columns:
    raise ValueError("A coluna 'original_poem' não foi encontrada no CSV.")

# Função para traduzir um poema com barra de progresso por estrofe
def traduzir_texto(texto):
    try:
        # Dividir o texto em estrofes/versos
        estrofes = texto.split('\n')
        
        traducao_completa = []
        for estrofe in tqdm(estrofes, desc="Traduzindo estrofes", leave=False):
            # Adicionar o prefixo de idioma ao texto de entrada
            estrofe_com_prefixo = f">>pt<< {estrofe}"
            
            # Tokenizar a estrofe
            encoded = tokenizer(estrofe_com_prefixo, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
            
            # Gerar a tradução
            with torch.no_grad():
                generated_tokens = model.generate(
                    **encoded,
                    max_length=512,  # Defina explicitamente o max_length
                    num_beams=5,  # Usar busca em feixe para melhorar a qualidade
                    early_stopping=True  # Parar a geração quando o modelo estiver confiante
                )
            
            # Decodificar a tradução
            traducao_completa.append(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
        
        # Juntar as traduções das estrofes em um único texto
        return '\n'.join(traducao_completa)
    except Exception as e:
        print(f"Erro ao traduzir o texto: {texto}. Erro: {e}")
        return ""  # Retorna uma string vazia em caso de erro

# Ativar tqdm no nível do dataframe
tqdm.pandas(desc="Traduzindo poemas...")

# Aplicar a tradução com progresso por poema
df["translated_by_marian"] = df["original_poem"].progress_apply(traduzir_texto)

# Verificar se todas as traduções estão vazias
if df["translated_by_marian"].empty or df["translated_by_marian"].isna().all():
    print("Aviso: Nenhuma tradução foi gerada. Verifique o texto de entrada e o modelo.")

# Salvar em um novo CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Tradução concluída! Arquivo salvo como {output_file}")