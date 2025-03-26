import os
import pandas as pd
from subword_nmt import learn_bpe
from subword_nmt.apply_bpe import BPE
import sacremoses
from tqdm import tqdm
import subprocess

# Caminhos dos arquivos CSV
CSV_TRAIN_PATH = "../poemas/poemas300/train/ingles_frances_train.csv"
CSV_VALID_PATH = "../poemas/poemas300/validation/ingles_frances_validation.csv"

# Função para pré-processamento
def preprocess_csv(csv_path, output_dir, prefix):
    """Processa o CSV e cria arquivos de treinamento/validação em formato .src e .tgt"""
    df = pd.read_csv(csv_path)
    
    src_lines = df['original_poem'].tolist()
    tgt_lines = df['translated_poem'].tolist()
    
    with open(f"{output_dir}/{prefix}.src", "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in src_lines)
    
    with open(f"{output_dir}/{prefix}.tgt", "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in tgt_lines)
    
    print(f"Arquivos {prefix}.src e {prefix}.tgt criados com sucesso!")

# Função para tokenização
def tokenize_file(input_file, output_file, lang):
    """Tokeniza os arquivos com a ferramenta Moses Tokenizer"""
    tokenizer = sacremoses.MosesTokenizer(lang=lang)
    
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"Tokenizando {lang}"):
            f_out.write(tokenizer.tokenize(line.strip(), return_str=True) + "\n")
    
    print(f"Arquivo {output_file} tokenizado com sucesso!")

# Função para aplicar BPE nos arquivos de dados
def apply_bpe_on_data(input_file, output_file, bpe_codes_file):
    """Aplica o BPE nos dados tokenizados e cria arquivos codificados"""
    with open(bpe_codes_file, "r", encoding="utf-8") as f_codes:
        bpe = BPE(f_codes)
    
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(bpe.process_line(line.strip()) + "\n")
    
    print(f"Arquivo {output_file} com BPE aplicado com sucesso!")

# Criar arquivo de configuração
def create_config():
    config_content = f"""
data:
    corpus_1:
        path_src: output_data/train.src.bpe
        path_tgt: output_data/train.tgt.bpe
    valid:
        path_src: output_data/valid.src.bpe
        path_tgt: output_data/valid.tgt.bpe

src_vocab: output_data/vocab.src
tgt_vocab: output_data/vocab.tgt

model_dir: output_model
save_model: output_model/model
save_data: output_data/onmt
save_checkpoint_steps: 1000
train_steps: 50000
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
learning_rate: 0.001
"""
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("Arquivo config.yaml criado com sucesso!")

# Criar vocabulário
def build_vocab():
    print("Criando vocabulário...")
    subprocess.run(["onmt_build_vocab", "-config", "config.yaml", "-n_sample", "-1"], check=True)
    print("Vocabulário criado com sucesso!")

# Treinar modelo
def train_model():
    print("Iniciando treinamento do modelo...")
    subprocess.run(["onmt_train", "-config", "config.yaml"], check=True)
    print("Treinamento concluído!")

# Função principal
def run():
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Processar os arquivos de treino e validação
    preprocess_csv(CSV_TRAIN_PATH, output_dir, "train")
    preprocess_csv(CSV_VALID_PATH, output_dir, "valid")
    
    # Tokenizar os arquivos
    tokenize_file(f"{output_dir}/train.src", f"{output_dir}/train.src.tok", "en")
    tokenize_file(f"{output_dir}/train.tgt", f"{output_dir}/train.tgt.tok", "fr")
    tokenize_file(f"{output_dir}/valid.src", f"{output_dir}/valid.src.tok", "en")
    tokenize_file(f"{output_dir}/valid.tgt", f"{output_dir}/valid.tgt.tok", "fr")
    
    # Aprender e aplicar BPE
    learn_bpe_on_data(f"{output_dir}/train.src.tok", f"{output_dir}/train.tgt.tok", "bpe.codes")
    apply_bpe_on_data(f"{output_dir}/train.src.tok", f"{output_dir}/train.src.bpe", "bpe.codes")
    apply_bpe_on_data(f"{output_dir}/train.tgt.tok", f"{output_dir}/train.tgt.bpe", "bpe.codes")
    apply_bpe_on_data(f"{output_dir}/valid.src.tok", f"{output_dir}/valid.src.bpe", "bpe.codes")
    apply_bpe_on_data(f"{output_dir}/valid.tgt.tok", f"{output_dir}/valid.tgt.bpe", "bpe.codes")
    
    create_config()
    build_vocab()
    train_model()

if __name__ == "__main__":
    run()
