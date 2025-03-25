import pandas as pd
import torch
from onmt.model_builder import load_test_model
from onmt.translate import Translator
import os
import argparse

# Configurações
CSV_INPUT = "../poemas/poemas300/test/frances_ingles_test2.csv"
CSV_OUTPUT = "../poemas/poemas300/openRNN/frances_ingles_poems_openRNN.csv"
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"

def load_translator_directly(model_path):
    """Carrega o modelo diretamente, contornando a API padrão"""
    try:
        # Configuração manual dos argumentos
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--src', type=str, default='en')
        parser.add_argument('--tgt', type=str, default='fr')
        parser.add_argument('--beam_size', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--gpu', type=int, default=0 if torch.cuda.is_available() else -1)
        
        args = parser.parse_args(['--model', model_path])
        
        # Carrega o modelo e os campos
        fields, model, model_opt = load_test_model(args)
        
        # Cria o tradutor
        translator = Translator.from_opt(
            opt=args,
            model=model,
            fields=fields,
            model_opt=model_opt
        )
        
        return translator
    except Exception as e:
        print(f"Erro detalhado no carregamento direto: {str(e)}")
        return None

def translate_texts(texts, translator):
    """Traduz uma lista de textos"""
    if translator is None:
        return ["ERRO: Tradutor não carregado"] * len(texts)
    
    translations = []
    for text in texts:
        try:
            result = translator.translate([text], batch_size=1)[0][0]
            translations.append(result)
        except Exception as e:
            print(f"Erro ao traduzir: {str(e)}")
            translations.append("ERRO NA TRADUÇÃO")
    return translations

def main():
    # Verificação de arquivo
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        print("Execute o treinamento primeiro ou verifique o caminho")
        return

    try:
        # Carregar dados
        df = pd.read_csv(CSV_INPUT)
        required_cols = {'original_poem', 'src_lang', 'tgt_lang'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Erro: CSV está faltando colunas: {missing}")
            return
        
        # Carregar modelo
        print(f"Carregando modelo diretamente: {MODEL_PATH}")
        translator = load_translator_directly(MODEL_PATH)
        
        if translator is None:
            print("Falha crítica no carregamento do modelo")
            print("Sugestões:")
            print("1. Verifique se o modelo foi gerado com a mesma versão do OpenNMT-py")
            print("2. Tente reinstalar o OpenNMT-py: pip install --force-reinstall opennmt-py")
            print("3. Considere retreinar o modelo com a versão atual")
            return
        
        # Tradução
        print("Traduzindo poemas...")
        df['translated_by_TA'] = translate_texts(df['original_poem'].tolist(), translator)
        
        # Salvar
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"Traduções salvas com sucesso em: {CSV_OUTPUT}")
        
    except Exception as e:
        print(f"Erro inesperado: {str(e)}")

if __name__ == "__main__":
    # Verificação de instalação
    try:
        from onmt.model_builder import load_test_model
    except ImportError:
        print("Erro: OpenNMT-py não está instalado corretamente")
        print("Instale com: pip install opennmt-py")
        exit(1)
    
    main()