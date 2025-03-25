import pandas as pd
import torch
from onmt.translate import Translator
from onmt.opts import translate_opts
import argparse
import os

# Configurações
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"

def load_rnn_model(model_path):
    """Carrega um modelo RNN do OpenNMT-py"""
    parser = argparse.ArgumentParser()
    translate_opts(parser)
    
    # Configuração completa para modelo RNN en→fr
    args = [
        '--model', model_path,
        '--src', 'en',                 # Idioma fonte
        '--tgt', 'fr',                 # Idioma alvo
        '--encoder_type', 'rnn',
        '--decoder_type', 'rnn',
        '--beam_size', '5',
        '--batch_size', '16',
        '--gpu', '0' if torch.cuda.is_available() else '-1'
    ]
    
    opt = parser.parse_args(args)
    translator = Translator.from_opt(opt)
    return translator

def translate_texts(texts, translator):
    """Traduz uma lista de textos"""
    return [translator.translate([text], batch_size=1)[0][0] for text in texts]

def main():
    # Verificar arquivos
    if not os.path.exists(MODEL_PATH):
        print(f"Modelo não encontrado: {MODEL_PATH}")
        return

    try:
        # Carregar dados
        df = pd.read_csv(CSV_INPUT)
        assert {'original_poem', 'src_lang', 'tgt_lang'}.issubset(df.columns)
        
        # Carregar modelo
        print(f"Carregando modelo: {MODEL_PATH}")
        translator = load_rnn_model(MODEL_PATH)
        
        # Traduzir
        print("Traduzindo poemas...")
        df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)
        
        # Salvar
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Traduções salvas em: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()