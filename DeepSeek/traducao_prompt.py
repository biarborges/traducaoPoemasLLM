from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
import os
from tqdm import tqdm

# Carrega modelo e tokenizer DeepSeek-R1 com código customizado
model_name = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)

# Cria pipeline SEM o argumento 'device'
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Caminhos
csv_path = "../traducaoPoemasLLM/poemas/frances_ingles_poems_teste.csv"
output_dir = "../traducaoPoemasLLM/poemas/deepseek"
output_path = f"{output_dir}/frances_ingles_poems_deepseek_prompt1_teste.csv"

if os.path.exists(csv_path):
    print("O diretório de entrada EXISTE.")
else:
    print("O diretório de entrada NÃO existe.")

if os.path.exists(output_dir):
    print("O diretório de saída EXISTE.")
else:
    print("O diretório de saída NÃO existe.")

# Lê o CSV
df = pd.read_csv(csv_path)

# Tradução com tqdm
tqdm.pandas()
def traduzir(poema):
    prompt = f"Traduza este poema do francês para o inglês:\n\n{poema}"
    response = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)[0]['generated_text']
    translated = response.replace(prompt, "").strip()
    return translated

# Aplica tradução
df["translated_by_TA"] = df["original_poem"].progress_apply(traduzir)

# Salva resultado
df.to_csv(output_path, index=False)
print(f"Traduções salvas em: {output_path}")
