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
    """Carrega o tradutor usando a API mais recente do OpenNMT-py"""
    try:
        # Configuração do servidor de tradução
        config = {
            'models': [
                {
                    'model': model_path,
                    'timeout': -1,
                    'on_timeout': 'to_cpu',
                    'load': True,
                    'tokenizer': {
                        'type': 'space'
                    }
                }
            ],
            'services': {
                'n_best': 1,
                'beam_size': 5,
                'batch_size': 16
            }
        }
        
        # Inicializa e carrega o modelo
        server = TranslationServer()
        server.start(config)
        
        # Retorna o primeiro modelo carregado
        return server.models[0][0]
    except Exception as e:
        print(f"Erro detalhado ao carregar modelo: {str(e)}")
        return None

def translate_texts(texts, translator):
    """Traduz uma lista de textos de forma robusta"""
    translations = []
    for text in texts:
        try:
            result = translator.translate([text])[0][0]
            translations.append(result)
        except Exception as e:
            print(f"Erro ao traduzir texto: {str(e)}")
            translations.append("ERRO NA TRADUÇÃO")
    return translations

def main():
    # Verificar se o modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Arquivo do modelo não encontrado em {MODEL_PATH}")
        print("Verifique se o caminho está correto e o modelo existe")
        return

    try:
        # Carregar dados
        df = pd.read_csv(CSV_INPUT)
        required_cols = {'original_poem', 'src_lang', 'tgt_lang'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Erro: Colunas faltando no CSV: {missing}")
            return
        
        # Carregar modelo
        print(f"Carregando modelo: {MODEL_PATH}")
        translator = load_translator(MODEL_PATH)
        
        if translator is None:
            print("Falha crítica: Não foi possível carregar o tradutor")
            print("Possíveis causas:")
            print("1. O arquivo do modelo está corrompido")
            print("2. Versão incompatível do OpenNMT-py")
            print("3. O modelo não é compatível com esta versão")
            return
        
        # Traduzir poemas
        print("Iniciando tradução dos poemas...")
        df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)
        
        # Salvar resultados
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Tradução concluída com sucesso! Resultados salvos em: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"Erro durante a execução: {str(e)}")

if __name__ == "__main__":
    # Verifica se o OpenNMT-py está instalado
    try:
        from onmt.translate import TranslationServer
    except ImportError:
        print("Erro: OpenNMT-py não está instalado corretamente")
        print("Instale com: pip install opennmt-py")
        exit(1)
    
    main()