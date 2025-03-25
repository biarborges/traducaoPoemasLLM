import pandas as pd
import os
import torch
from onmt.translate import TranslationServer, Translator

# Configurações
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"

# Configuração do modelo RNN
MODEL_CONFIG = {
    'model_path': MODEL_PATH,
    'src': 'en',
    'tgt': 'fr',
    'encoder_type': 'rnn',
    'decoder_type': 'rnn',
    'beam_size': 5,
    'batch_size': 16,
    'gpu': 0 if torch.cuda.is_available() else -1
}

def initialize_translator(config):
    """Inicializa o tradutor para modelos RNN"""
    server = TranslationServer()
    server.start(config)
    return server.models[0][0]  # Retorna o primeiro modelo carregado

def translate_texts(texts, translator):
    """Traduz uma lista de textos"""
    translations = []
    for text in texts:
        result = translator.translate([text])
        translations.append(result[0][0])  # Pega a primeira hipótese da primeira tradução
    return translations

def main():
    # Verificar se o modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return

    # Carregar dados
    try:
        df = pd.read_csv(CSV_INPUT)
        required_cols = {'original_poem', 'src_lang', 'tgt_lang'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Colunas faltando no CSV: {missing}")
            return
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        return

    # Inicializar tradutor
    try:
        print(f"Carregando modelo RNN: {MODEL_PATH}")
        translator = initialize_translator({'models': [MODEL_CONFIG]})
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    # Processar traduções
    try:
        print("Iniciando tradução...")
        df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)
        
        # Salvar resultados
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Traduções salvas em: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"Erro durante tradução: {e}")

if __name__ == "__main__":
    main()