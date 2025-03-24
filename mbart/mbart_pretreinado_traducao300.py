import os
import torch
import pandas as pd
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Forçar execução síncrona de CUDA para depuração
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

start_time = time.time()

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar o modelo e o tokenizador
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Carregar o arquivo CSV
input_csv = "../poemas/poemas300/test/frances_ingles_test.csv"
df = pd.read_csv(input_csv)

# Definir o idioma de origem para o francês (ou ajustar conforme o CSV)
tokenizer.src_lang = "fr_XX"

# Função para traduzir um poema
def traduzir_poema(poema):
    # Garantir que os tensores estejam no mesmo dispositivo
    encoded_text = tokenizer(poema, return_tensors="pt").to(device)
    
    # Depuração: Imprimir a entrada tokenizada
    print(f"Entrada tokenizada: {encoded_text}")

    # Gerar a tradução
    generated_tokens = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    
    # Decodificar a tradução
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

# Aplicar a função de tradução a todos os poemas e armazenar no novo campo 'translated_by_TA'
df['translated_by_TA'] = df['original_poem'].apply(traduzir_poema)

# Salvar o arquivo CSV com a nova coluna
output_csv = "../poemas/poemas300/test/frances_ingles_test_pretreinado_mbart.csv"
df.to_csv(output_csv, index=False)

# Exibir o resultado
print(f"Arquivo CSV com traduções salvo em: {output_csv}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total de execução: {elapsed_time:.2f} segundos")
