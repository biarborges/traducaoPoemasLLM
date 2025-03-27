import os
import pandas as pd
import sacremoses
import subprocess
import time
import multiprocessing
from subword_nmt import learn_bpe
from subword_nmt.apply_bpe import BPE

start_time = time.time()

# Caminhos dos arquivos CSV
CSV_TRAIN_PATH = "../poemas/poemas300/train/ingles_frances_train.csv"
CSV_VALID_PATH = "../poemas/poemas300/validation/ingles_frances_validation.csv"

# Diretório de saída
OUTPUT_DIR = "output_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função para pré-processamento
def preprocess_csv(csv_path, prefix):
    df = pd.read_csv(csv_path, usecols=['original_poem', 'translated_poem'])
    df['original_poem'].to_csv(f"{OUTPUT_DIR}/{prefix}.src", index=False, header=False)
    df['translated_poem'].to_csv(f"{OUTPUT_DIR}/{prefix}.tgt", index=False, header=False)
    print(f"Arquivos {prefix}.src e {prefix}.tgt criados!")

# Função para tokenização com multiprocessing
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

# Criar configuração
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

model_dir: output_model
save_model: output_model/model
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

# Criar vocabulário
def build_vocab():
    # Verificar e remover arquivos de vocabulário existentes
    vocab_src_path = f"{OUTPUT_DIR}/vocab.src"
    vocab_tgt_path = f"{OUTPUT_DIR}/vocab.tgt"
    
    if os.path.exists(vocab_src_path):
        os.remove(vocab_src_path)
        print(f"{vocab_src_path} removido.")
        
    if os.path.exists(vocab_tgt_path):
        os.remove(vocab_tgt_path)
        print(f"{vocab_tgt_path} removido.")
    
    # Criar vocabulário
    subprocess.run(["onmt_build_vocab", "-config", "config.yaml", "-n_sample", "-1"], check=True)
    print("Vocabulário criado!")

# Treinar modelo
def train_model():
    subprocess.run(["onmt_train", "-config", "config.yaml"], check=True)
    print("Treinamento concluído!")

# Função principal
def run():
    preprocess_csv(CSV_TRAIN_PATH, "train")
    preprocess_csv(CSV_VALID_PATH, "valid")
    
    # Tokenizar usando multiprocessing
    jobs = [
        (f"{OUTPUT_DIR}/train.src", f"{OUTPUT_DIR}/train.src.tok", "en"),
        (f"{OUTPUT_DIR}/train.tgt", f"{OUTPUT_DIR}/train.tgt.tok", "fr"),
        (f"{OUTPUT_DIR}/valid.src", f"{OUTPUT_DIR}/valid.src.tok", "en"),
        (f"{OUTPUT_DIR}/valid.tgt", f"{OUTPUT_DIR}/valid.tgt.tok", "fr"),
    ]
    with multiprocessing.Pool(len(jobs)) as pool:
        pool.starmap(tokenize_file, jobs)
    
    # Aprender e aplicar BPE
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
