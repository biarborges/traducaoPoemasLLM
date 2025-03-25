import pandas as pd
import os
import yaml
from onmt.translate import Translator
from onmt.model_builder import load_test_model

# Caminhos
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"  # Substituir pelo caminho correto
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"  # Ajuste conforme necessário
CONFIG_PATH = "config.yaml"

# Carregar configuração
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Traduzir usando modelo treinado
def translate_texts(texts, translator):
    translations = translator.translate(texts, batch_size=16)
    return [t[0] for t in translations]

# Carregar dados
df = pd.read_csv(CSV_INPUT)

# Verificar colunas necessárias
if not {'original_poem', 'src_lang', 'tgt_lang'}.issubset(df.columns):
    raise ValueError("O CSV deve conter as colunas 'original_poem', 'src_lang' e 'tgt_lang'")

# Carregar configuração
config = load_config(CONFIG_PATH)

# Ajustar o dicionário de configuração para carregar o modelo corretamente
opt = {
    'models': [MODEL_PATH],
    'src': config['data']['corpus_1']['path_src'],
    'tgt': config['data']['corpus_1']['path_tgt'],
    'batch_size': 16,
    'gpu': config.get('gpu_ranks', [0]),
    'beam_size': 5,
}

# Carregar o modelo de tradução
model, fields = load_test_model(opt)

# Inicializar o tradutor para RNN
translator = Translator(model=model, fields=fields, beam_size=5)

# Traduzir os poemas
df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)

# Salvar novo CSV
df.to_csv(CSV_OUTPUT, index=False)
print("Tradução concluída e salva em", CSV_OUTPUT)
