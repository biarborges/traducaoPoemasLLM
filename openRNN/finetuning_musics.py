import os
import pandas as pd
import sacremoses
import subprocess
import time
import torch
import multiprocessing
from subword_nmt import learn_bpe
from subword_nmt.apply_bpe import BPE

os.environ['TORCH_CUDA_VERSION'] = ''

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Caminhos dos arquivos CSV
CSV_POEM_TRAIN = "../poemas/train/portugues_ingles_train.csv"
CSV_POEM_VALID = "../poemas/validation/portugues_ingles_validation.csv"
CSV_SONG_TRAIN = "../musicas/train/portugues_ingles_musics_train.csv"
CSV_SONG_VALID = "../musicas/validation/portugues_ingles_musics_validation.csv"

# Diretório de saída
OUTPUT_DIR = "output_data_duplo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pré-processamento CSV para gerar arquivos .src e .tgt
def preprocess_csv(csv_path, prefix):
    df = pd.read_csv(csv_path, usecols=['original_poem', 'translated_poem'])
    df['original_poem'].to_csv(f"{OUTPUT_DIR}/{prefix}.src", index=False, header=False)
    df['translated_poem'].to_csv(f"{OUTPUT_DIR}/{prefix}.tgt", index=False, header=False)
    print(f"{prefix}.src e {prefix}.tgt criados!")

# Tokenização
def tokenize_file(input_file, output_file, lang):
    tokenizer = sacremoses.MosesTokenizer(lang=lang)
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(tokenizer.tokenize(line.strip(), return_str=True) + "\n" for line in f_in)
    print(f"{output_file} tokenizado!")

# Aprender BPE
def learn_bpe_on_data(src_file, tgt_file, bpe_codes_file):
    with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt, open(bpe_codes_file, "w", encoding="utf-8") as f_bpe:
        learn_bpe.learn_bpe(f_src.readlines() + f_tgt.readlines(), f_bpe, num_symbols=10000, min_frequency=2)
    print("BPE aprendido!")

# Aplicar BPE
def apply_bpe_on_data(input_file, output_file, bpe_codes_file):
    with open(bpe_codes_file, "r", encoding="utf-8") as f_codes, open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        bpe = BPE(f_codes)
        f_out.writelines(bpe.process_line(line.strip()) + "\n" for line in f_in)
    print(f"{output_file} com BPE aplicado!")

# Criar config
def create_config():
    config_content = f"""
data:
    corpus_1:
        path_src: {OUTPUT_DIR}/train.src.bpe
        path_tgt: {OUTPUT_DIR}/train.tgt.bpe
    valid:
        path_src: {OUTPUT_DIR}/valid.src.bpe
        path_tgt: {OUTPUT_DIR}/valid.tgt.bpe

src_vocab: {OUTPUT_DIR}/vocab.src
tgt_vocab: {OUTPUT_DIR}/vocab.tgt

model_dir: output_model_duplo
save_model: output_model_duplo/model
save_data: {OUTPUT_DIR}/onmt
save_checkpoint_steps: 1000
train_steps: 10000
valid_steps: 1000
world_size: 1
gpu_ranks: [0]

encoder_type: brnn
decoder_type: rnn
rnn_size: 512
word_vec_size: 300
layers: 2
dropout: 0.3
optim: adam
learning_rate: 0.002
fp16: true
"""
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("config.yaml criado!")

# Construir vocabulário
def build_vocab():
    # First build vocab files with word counts
    for lang in ['src', 'tgt']:
        with open(f"{OUTPUT_DIR}/train.{lang}.bpe", "r") as f_in:
            words = [word for line in f_in for word in line.strip().split()]
            from collections import Counter
            word_counts = Counter(words)
            
        with open(f"{OUTPUT_DIR}/vocab.{lang}", "w") as f_out:
            for word, count in word_counts.most_common():
                f_out.write(f"{word} {count}\n")
    
    print("Vocabulário criado com contagens!")

# Treinar
def train_model():
    subprocess.run(["onmt_train", "-config", "config.yaml"], check=True)
    print("Treinamento concluído!")

# Função principal
def run():
    # Pré-processar poemas e músicas
    preprocess_csv(CSV_POEM_TRAIN, "poem_train")
    preprocess_csv(CSV_POEM_VALID, "poem_valid")
    preprocess_csv(CSV_SONG_TRAIN, "song_train")
    preprocess_csv(CSV_SONG_VALID, "song_valid")

    # Combinar os arquivos de treino e validação
    def combine_files(file1, file2, out_file):
        with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2, open(out_file, "w", encoding="utf-8") as fout:
            fout.writelines(f1.readlines() + f2.readlines())
        print(f"{out_file} combinado.")

    combine_files(f"{OUTPUT_DIR}/poem_train.src", f"{OUTPUT_DIR}/song_train.src", f"{OUTPUT_DIR}/train.src")
    combine_files(f"{OUTPUT_DIR}/poem_train.tgt", f"{OUTPUT_DIR}/song_train.tgt", f"{OUTPUT_DIR}/train.tgt")
    combine_files(f"{OUTPUT_DIR}/poem_valid.src", f"{OUTPUT_DIR}/song_valid.src", f"{OUTPUT_DIR}/valid.src")
    combine_files(f"{OUTPUT_DIR}/poem_valid.tgt", f"{OUTPUT_DIR}/song_valid.tgt", f"{OUTPUT_DIR}/valid.tgt")

    # Tokenizar com multiprocessing
    jobs = [
        (f"{OUTPUT_DIR}/train.src", f"{OUTPUT_DIR}/train.src.tok", "pt"),
        (f"{OUTPUT_DIR}/train.tgt", f"{OUTPUT_DIR}/train.tgt.tok", "en"),
        (f"{OUTPUT_DIR}/valid.src", f"{OUTPUT_DIR}/valid.src.tok", "pt"),
        (f"{OUTPUT_DIR}/valid.tgt", f"{OUTPUT_DIR}/valid.tgt.tok", "en"),
    ]
    with multiprocessing.Pool(len(jobs)) as pool:
        pool.starmap(tokenize_file, jobs)

    learn_bpe_on_data(f"{OUTPUT_DIR}/train.src.tok", f"{OUTPUT_DIR}/train.tgt.tok", "bpe.codes")

    bpe_jobs = [
        (f"{OUTPUT_DIR}/train.src.tok", f"{OUTPUT_DIR}/train.src.bpe", "bpe.codes"),
        (f"{OUTPUT_DIR}/train.tgt.tok", f"{OUTPUT_DIR}/train.tgt.bpe", "bpe.codes"),
        (f"{OUTPUT_DIR}/valid.src.tok", f"{OUTPUT_DIR}/valid.src.bpe", "bpe.codes"),
        (f"{OUTPUT_DIR}/valid.tgt.tok", f"{OUTPUT_DIR}/valid.tgt.bpe", "bpe.codes"),
    ]
    with multiprocessing.Pool(len(bpe_jobs)) as pool:
        pool.starmap(apply_bpe_on_data, bpe_jobs)

    create_config()
    build_vocab()
    train_model()

if __name__ == "__main__":
    run()
    print(f"Tempo total: {time.time() - start_time:.2f} segundos")
