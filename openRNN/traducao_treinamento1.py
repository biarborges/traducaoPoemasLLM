import pandas as pd
import os
import yaml
from onmt.translate import Translator
from onmt.model_builder import load_test_model
import torch

# Caminhos
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"

# Configuração manual (substitui o arquivo YAML)
CONFIG = {
    'models': [MODEL_PATH],
    'src': 'en',  # Idioma fonte
    'tgt': 'fr',  # Idioma alvo
    'beam_size': 5,
    'batch_size': 16,
    'gpu': 0 if torch.cuda.is_available() else -1,
    'fp32': True,
    'encoder_type': 'lstm',
    'decoder_type': 'lstm'
}

def translate_texts(texts, translator):
    """Traduz uma lista de textos usando o modelo carregado"""
    return [translation[0] for translation in translator.translate(texts, batch_size=CONFIG['batch_size'])]

def main():
    # Carregar dados
    try:
        df = pd.read_csv(CSV_INPUT)
        if not {'original_poem', 'src_lang', 'tgt_lang'}.issubset(df.columns):
            raise ValueError("CSV deve conter colunas: 'original_poem', 'src_lang', 'tgt_lang'")
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        return

    # Carregar modelo
    try:
        print(f"Carregando modelo: {MODEL_PATH}")
        model, fields = load_test_model(CONFIG)
        translator = Translator.from_config(
            CONFIG,
            model,
            fields,
            beam_size=CONFIG['beam_size']
        )
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    # Traduzir poemas
    try:
        print("Iniciando tradução...")
        df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)
        
        # Salvar resultados
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Tradução concluída! Resultados salvos em: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"Erro durante tradução: {e}")

if __name__ == "__main__":
    import torch  # Import necessário para verificação de GPU
    main()