import csv
import os
from tqdm import tqdm
import torch
from onmt.translate import Translator
from onmt.utils.parse import ArgumentParser

# Configurações
MODEL_PATH = "models_fr_en/model_en_fr"  # Caminho para o modelo treinado
SRC_LANG = "fr"  # Idioma de origem
TGT_LANG = "en"  # Idioma de destino
INPUT_FILE = os.path.abspath("../poemas/poemas300/test/frances_ingles_test2.csv")  # Arquivo CSV de entrada
OUTPUT_FILE = os.path.abspath("../poemas/poemas300/openRNN/frances_ingles_poems_openrnn.csv")  # Arquivo CSV de saída

# Função para carregar o modelo
def load_model(model_path, gpu=True):
    parser = ArgumentParser()
    opt = parser.parse_args(["-model", model_path, "-gpu", "0" if gpu else "-1"])
    translator = Translator.from_opt(opt)
    return translator

# Função para traduzir o poema
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
            src_text = original_poem.strip().split('\n')
            translated_text = translator.translate(src_text)

            # Escreve o poema traduzido no CSV de saída
            row['translated_poem'] = '\n'.join(translated_text)
            row['translated_by_TA'] = "OpenRNN"  # Adiciona a coluna 'translated_by_TA'
            
            writer.writerow(row)

# Função principal
def main():
    # Carregar o modelo treinado
    translator = load_model(MODEL_PATH, gpu=True)

    # Traduzir os poemas e salvar no novo CSV
    translate_poems(INPUT_FILE, OUTPUT_FILE, translator)
    print(f"Tradução concluída e salva em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
