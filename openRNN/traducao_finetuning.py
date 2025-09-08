import pandas as pd
import torch
import onmt.translate
import onmt.opts
import onmt.model_builder
from onmt.utils.parse import ArgumentParser
import os

# --- 1. DEFINIR CAMINHOS E PARÂMETROS ---
print("Passo 1: Definindo caminhos e parâmetros...")
BASE_DIR = r"C:\Users\biarb\OneDrive\UFU\Mestrado\Dissertacao\traducaoPoemasLLM"
INPUT_FILE = os.path.join(BASE_DIR, "poemas", "test", "frances_ingles_test.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "poemas", "openRNN", "finetuning_musics", "frances_ingles.csv")
MODEL_PATH = os.path.join(BASE_DIR, "openRNN", "model_step_10000.pt")
GPU = -1
BEAM_SIZE = 5

# --- 2. VERIFICAÇÃO DOS ARQUIVOS ---
print("\nPasso 2: Verificando a existência dos arquivos necessários...")
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Arquivo de entrada não encontrado: {INPUT_FILE}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
print("-> Arquivos encontrados.")

# --- 3. CARREGAR E PREPARAR OS DADOS ---
print("\nPasso 3: Lendo poemas do arquivo CSV...")
df = pd.read_csv(INPUT_FILE)
poems_to_translate = df['original_poem'].tolist()
poems_to_translate = [str(poem) for poem in poems_to_translate]
print(f"-> Encontrados {len(poems_to_translate)} poemas para traduzir.")

# --- 4. CARREGAR CHECKPOINT E MODIFICAR OPÇÕES ---
print("\nPasso 4: Carregando o checkpoint do modelo...")
device = torch.device("cpu")
# NOTA: Se você voltou para a versão só-CPU do PyTorch, mantenha. Se ainda está com a versão CUDA, mantenha também.
# O código abaixo funcionará em ambos os casos.
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
fields = checkpoint['vocab']
model_opt = checkpoint['opt']

# --- A CIRURGIA - A SOLUÇÃO DEFINITIVA ---
# Nós modificamos as opções carregadas para forçar o modo CPU,
# apagando a configuração de GPU com a qual o modelo foi treinado.
print("-> Modificando opções do modelo para forçar o uso da CPU...")
model_opt.gpu_ranks = []
model_opt.world_size = 1
# -----------------------------------------

print("-> Construindo o modelo com as opções modificadas...")
model = onmt.model_builder.build_base_model(model_opt, fields, GPU, checkpoint=checkpoint)
model.eval() 
print("-> Modelo construído e carregado com sucesso em modo CPU.")

# --- 5. CONFIGURAR E EXECUTAR O TRADUTOR ---
def translate_poems(poems_list, beam_size, gpu, built_model, model_config, vocab_fields):
    parser = ArgumentParser()
    onmt.opts.config_opts(parser)
    onmt.opts.translate_opts(parser)
    args = ['-model', MODEL_PATH, '-src', "dummy_src", '-gpu', str(gpu), '-beam_size', str(beam_size), '-n_best', '1']
    opt = parser.parse_args(args)
    translator = onmt.translate.Translator.from_opt(model=built_model, fields=vocab_fields, opt=opt, model_opt=model_config)
    scores, predictions = translator.translate(src=poems_list, batch_size=32)
    translated_texts = [pred[0] for pred in predictions]
    return translated_texts

print("\nPasso 5: Iniciando a tradução...")
translations = translate_poems(poems_to_translate, BEAM_SIZE, GPU, model, model_opt, fields)
print("-> Tradução concluída com sucesso!")

# --- 6. SALVAR O RESULTADO ---
print("\nPasso 6: Salvando o resultado...")
df['translated_by_TA'] = translations
output_dir = os.path.dirname(OUTPUT_FILE)
os.makedirs(output_dir, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"\nPROCESSO FINALIZADO COM SUCESSO! \nO arquivo com as traduções foi salvo em:\n{OUTPUT_FILE}")