import csv
import os
import argparse
from tqdm import tqdm
import torch
import onmt.translate
from onmt.translate.translator import build_translator

# Configurações
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"  # Caminho do modelo treinado
SRC_LANG = "fr"
TGT_LANG = "en"
INPUT_FILE = os.path.abspath("../poemas/poemas300/test/frances_ingles_test2.csv")
OUTPUT_FILE = os.path.abspath("../poemas/poemas300/openRNN/frances_ingles_poems_openrnn.csv")
TEMP_OUTPUT_FILE = os.path.abspath("temp_output.txt")  # Arquivo temporário exigido pelo OpenNMT

# Função para criar argumentos do tradutor
def get_translator_options(model_path, gpu=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("-models", type=str, nargs="+", default=[model_path])
    parser.add_argument("-gpu", type=int, default=0 if gpu and torch.cuda.is_available() else -1)
    parser.add_argument("-gpu_ranks", type=int, nargs="+", default=[0] if gpu and torch.cuda.is_available() else [])
    parser.add_argument("-src", type=str, default=None)
    parser.add_argument("-tgt", type=str, default=None)
    parser.add_argument("-output", type=str, default=TEMP_OUTPUT_FILE)
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-replace_unk", action="store_true", default=True)
    parser.add_argument("-verbose", action="store_true", default=False)
    parser.add_argument("-world_size", type=int, default=1)
    parser.add_argument("-parallel_mode", type=str, default="data_parallel")
    parser.add_argument("-precision", type=str, default="fp32")

    # Parâmetros adicionais
    parser.add_argument("-alpha", type=float, default=0.0)  
    parser.add_argument("-beta", type=float, default=0.0)  
    parser.add_argument("-length_penalty", type=str, default="none")
    parser.add_argument("-coverage_penalty", type=str, default="none")
    parser.add_argument("-report_align", action="store_true", default=False)
    parser.add_argument("-n_best", type=int, default=1)  # Retorna apenas a melhor tradução
    parser.add_argument("-min_length", type=int, default=1)  # Definindo o comprimento mínimo da tradução
    parser.add_argument("-max_length", type=int, default=100)  # Definindo o comprimento máximo da tradução
    parser.add_argument("-max_length_ratio", type=float, default=1.0)  # Proporção do comprimento máximo

    return parser.parse_args([])  # Retorna um Namespace

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
        fieldnames = reader.fieldnames + ['translated_by_TA']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()

        for row in tqdm(reader, desc="Traduzindo poemas"):
            original_poem = row['original_poem']
            translated_text = translator.translate([original_poem])[0]

            row['translated_poem'] = translated_text
            row['translated_by_TA'] = "OpenRNN"
            
            writer.writerow(row)

# Função principal
def main():
    translator = load_translator(MODEL_PATH, gpu=True)
    translate_poems(INPUT_FILE, OUTPUT_FILE, translator)
    print(f"Tradução concluída e salva em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
