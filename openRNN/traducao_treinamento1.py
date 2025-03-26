import csv
import os
import argparse
from tqdm import tqdm
import torch
import onmt.translate
from onmt.translate.translator import build_translator
from onmt.model_builder import load_test_model

# Configurações
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"  # Caminho do modelo treinado (verifique se está correto)
SRC_LANG = "fr"  # Idioma de origem
TGT_LANG = "en"  # Idioma de destino
INPUT_FILE = os.path.abspath("../poemas/poemas300/test/frances_ingles_test2.csv")  # Arquivo CSV de entrada
OUTPUT_FILE = os.path.abspath("../poemas/poemas300/openRNN/frances_ingles_poems_openrnn.csv")  # Arquivo CSV de saída
TEMP_OUTPUT_FILE = os.path.abspath("temp_output.txt")  # Arquivo temporário exigido pelo OpenNMT

# Função para criar um objeto de argumentos (Namespace)
def get_translator_options(model_path, gpu=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-models", type=str, nargs="+", default=[model_path])  # Corrigido para uma lista
    parser.add_argument("-gpu", type=int, default=0 if gpu and torch.cuda.is_available() else -1)
    parser.add_argument("-src", type=str, default=None)
    parser.add_argument("-tgt", type=str, default=None)
    parser.add_argument("-output", type=str, default=TEMP_OUTPUT_FILE)  # Define um arquivo temporário de saída
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-replace_unk", action="store_true", default=True)
    parser.add_argument("-verbose", action="store_true", default=False)
    parser.add_argument("-world_size", type=int, default=1)  # Corrigido: Adicionando world_size
    parser.add_argument("-parallel_mode", type=str, default="data_parallel")  # Corrigido: Adicionando parallel_mode
    
    return parser.parse_args([])  # Retorna um objeto Namespace sem argumentos da linha de comando

# Função para carregar o modelo corretamente
def load_translator(model_path, gpu=True):
    opt = get_translator_options(model_path, gpu)
    translator = build_translator(opt, report_score=False)
    return translator

# Função para traduzir os poemas
def translate_poems(input_file, output_file, translator):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['translated_by_TA']  # Adiciona a nova coluna
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()

        for row in tqdm(reader, desc="Traduzindo poemas"):
            original_poem = row['original_poem']
            src_lang = row['src_lang']
            tgt_lang = row['tgt_lang']

            # Traduz o poema
            translated_text = translator.translate([original_poem])[0]

            # Escreve o poema traduzido no CSV de saída
            row['translated_poem'] = translated_text
            row['translated_by_TA'] = "OpenRNN"  # Adiciona a coluna 'translated_by_TA'
            
            writer.writerow(row)

# Função principal
def main():
    # Carregar o modelo treinado
    translator = load_translator(MODEL_PATH, gpu=True)

    # Traduzir os poemas e salvar no novo CSV
    translate_poems(INPUT_FILE, OUTPUT_FILE, translator)
    print(f"Tradução concluída e salva em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
