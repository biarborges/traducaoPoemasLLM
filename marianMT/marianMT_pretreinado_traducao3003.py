#esse daqui pareceu melhor nos resultados visuais, traduziu todas as frases apesar de muitas terem repetiçoes, mas traduziu tudo!!! 0 300 nao traduziu tudo, mas teve resultados melhores nas métricas. o 3002 é pra ver o poema sozinho.
import torch
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do MarianMT
#model_name = "/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-160"
model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"  
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Função para traduzir poema
def traduzir_poema(poema, tokenizer, model, device):
    versos = poema.split("\n")
    traducao_completa = []

    # Traduzir verso por verso
    for verso in versos:
        texto_com_prefixo = f">>fr<< {verso.strip()}"  # Adicionar prefixo da língua
        encoded = tokenizer(texto_com_prefixo, return_tensors="pt", truncation=True, padding=True, max_length=512)
        encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU

        with torch.no_grad():
            generated_tokens = model.generate(**encoded, max_length=512, num_beams=5)

        traducao = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        traducao_completa.append(traducao)

    return "\n".join(traducao_completa)

# Carregar o CSV com os poemas
df = pd.read_csv('../poemas/poemas300/test/ingles_frances_test.csv')

# Adicionar a coluna para as traduções
df['translated_by_marian'] = df['original_poem'].apply(lambda x: traduzir_poema(x, tokenizer, model, device))

# Salvar o CSV com a tradução
df.to_csv('../poemas/poemas300/marianmt/ingles_frances_test_pretreinado_marianmt.csv', index=False)

print("Tradução concluída e salva.")
