import pandas as pd
import torch
from onmt.translate import TranslationServer
import os
import yaml

# Configurações
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"

def load_translator(model_path):
    """Carrega o tradutor usando a API mais recente"""
    # Configuração do modelo
    config = {
        'models': [{
            'model_path': model_path,
            'timeout': -1,
            'on_timeout': 'to_cpu',
            'load': True,
            'tokenizer': {
                'type': 'space'
            }
        }],
        'services': {
            'n_best': 1,
            'beam_size': 5,
            'batch_size': 16
        }
    }
    
    # Inicializa o servidor de tradução
    server = TranslationServer()
    server.start(config)
    
    # Retorna o primeiro modelo carregado
    return server.models[0][0]

def translate_texts(texts, translator):
    """Traduz uma lista de textos"""
    translations = []
    for text in texts:
        result = translator.translate([text])[0][0]
        translations.append(result)
    return translations

def main():
    # Verificar arquivos
    if not os.path.exists(MODEL_PATH):
        print(f"Modelo não encontrado: {MODEL_PATH}")
        return

    try:
        # Carregar dados
        df = pd.read_csv(CSV_INPUT)
        required_cols = {'original_poem', 'src_lang', 'tgt_lang'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Colunas faltando: {missing}")
            return
        
        # Carregar modelo
        print(f"Carregando modelo: {MODEL_PATH}")
        translator = load_translator(MODEL_PATH)
        
        # Traduzir
        print("Traduzindo poemas...")
        df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)
        
        # Salvar
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Traduções salvas em: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()