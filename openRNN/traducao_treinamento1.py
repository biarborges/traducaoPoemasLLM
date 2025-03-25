import pandas as pd
import torch
from onmt.translate import Translator
from onmt.utils.parse import ArgumentParser
import os

# Configurações
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"

def load_rnn_translator(model_path):
    """Carrega um tradutor para modelo RNN com a API atual"""
    # Configuração do parser
    parser = ArgumentParser()
    group = parser.add_argument_group('Translation')
    group.add_argument('--model', required=True, help="Path to model .pt file")
    group.add_argument('--src', default="en", help="Source language")
    group.add_argument('--tgt', default="fr", help="Target language")
    group.add_argument('--beam_size', type=int, default=5)
    group.add_argument('--batch_size', type=int, default=16)
    group.add_argument('--gpu', type=int, default=0 if torch.cuda.is_available() else -1)
    
    # Parse dos argumentos
    opt = parser.parse_args([
        '--model', model_path,
        '--src', 'en',
        '--tgt', 'fr'
    ])
    
    # Carrega o tradutor
    return Translator.from_opt(opt)

def translate_texts(texts, translator):
    """Traduz uma lista de textos"""
    translations = []
    for text in texts:
        # Para cada texto, faz a tradução individualmente
        translation = translator.translate([text], batch_size=1)[0][0]
        translations.append(translation)
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
        translator = load_rnn_translator(MODEL_PATH)
        
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