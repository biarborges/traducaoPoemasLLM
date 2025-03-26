import csv
import os
import torch
import onmt.translate
from onmt.translate.translator import build_translator
from argparse import Namespace
from tqdm import tqdm

# Configurações
MODEL_PATH = "../openRNN/models_en_fr/model_en_fr_step_50000.pt"  # Caminho do modelo treinado
SRC_LANG = "fr"
TGT_LANG = "en"
INPUT_FILE = os.path.abspath("../poemas/poemas300/test/frances_ingles_test2.csv")
OUTPUT_FILE = os.path.abspath("../poemas/poemas300/openRNN/frances_ingles_poems_openrnn.csv")
TEMP_OUTPUT_FILE = os.path.abspath("temp_output.txt")  # Arquivo temporário exigido pelo OpenNMT

# Função para criar argumentos do tradutor
def get_translator_options(model_path, gpu=True):
    opt = Namespace(
        models=[model_path],
        gpu=0 if gpu and torch.cuda.is_available() else -1,
        gpu_ranks=[0] if gpu and torch.cuda.is_available() else [],
        src=None, tgt=None, output=TEMP_OUTPUT_FILE,
        batch_size=1, replace_unk=True, verbose=False,
        world_size=1, parallel_mode="data_parallel", precision="fp32",
        alpha=0.0, beta=0.0, length_penalty="none", coverage_penalty="none",
        report_align=False, n_best=1, min_length=1, max_length=100,
        max_length_ratio=1.0, ratio=1.0, beam_size=5, random_sampling_topk=10
    )
    return opt

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
        fieldnames = reader.fieldnames + ['translated_poem', 'translated_by_TA']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Traduzindo poemas"):
            original_poem = row['original_poem']
            
            # Escreve o poema em um arquivo temporário
            with open(TEMP_OUTPUT_FILE, 'w', encoding='utf-8') as temp_src:
                temp_src.write(original_poem + '\n')

            # Executa a tradução usando o OpenNMT
            translator.translate(
                src_path=TEMP_OUTPUT_FILE,
                tgt_path=None,
                src=[original_poem]
            )

            # Lê a saída gerada pelo OpenNMT
            with open(TEMP_OUTPUT_FILE, 'r', encoding='utf-8') as temp_out:
                translated_text = temp_out.read().strip()

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
