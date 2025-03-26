import os
import pandas as pd
from subword_nmt import learn_bpe
from subword_nmt.apply_bpe import BPE
import sacremoses
from tqdm import tqdm
import subprocess

# Caminhos dos arquivos CSV
CSV_TRAIN_PATH = "../poemas/poemas300/train/ingles_frances_train.csv"  # Arquivo de treinamento
CSV_VALID_PATH = "../poemas/poemas300/validation/ingles_frances_validation.csv"  # Arquivo de validação

# Função para pré-processamento
def preprocess_csv(csv_path, output_dir):
    """Processa o CSV e cria arquivos de treinamento e validação em formato .src e .tgt"""
    df = pd.read_csv(csv_path)
    
    src_lines = []
    tgt_lines = []
    
    for _, row in df.iterrows():
        src_lines.append(row['original_poem'])
        tgt_lines.append(row['translated_poem'])

    # Salvando como arquivos .src e .tgt
    with open(f"{output_dir}/train.src", "w", encoding="utf-8") as f:
        for line in src_lines:
            f.write(line + "\n")
            
    with open(f"{output_dir}/train.tgt", "w", encoding="utf-8") as f:
        for line in tgt_lines:
            f.write(line + "\n")
            
    print("Arquivos train.src e train.tgt criados com sucesso!")

# Função para tokenização
def tokenize_file(input_file, output_file, lang):
    """Tokeniza os arquivos com a ferramenta Moses Tokenizer"""
    tokenizer = sacremoses.MosesTokenizer(lang=lang)
    
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"Tokenizando {lang}"):
            f_out.write(tokenizer.tokenize(line.strip(), return_str=True) + "\n")
    
    print(f"Arquivo {output_file} tokenizado com sucesso!")

# Função para aprender o BPE
def learn_bpe_on_data(src_file, tgt_file, bpe_codes_file):
    """Aprende o BPE a partir dos arquivos tokenizados"""
    src_lines = []
    tgt_lines = []
    
    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f.readlines()]
        
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f.readlines()]

    # Aplicando o BPE separadamente para src e tgt
    with open(bpe_codes_file, "w", encoding="utf-8") as f_bpe:
        learn_bpe.learn_bpe(src_lines + tgt_lines, f_bpe, num_symbols=10000, min_frequency=2)
        
    print("Aprendizado BPE concluído!")

# Função para aplicar BPE nos arquivos de dados
def apply_bpe_on_data(src_file, tgt_file, bpe_codes_file, output_dir):
    """Aplica o BPE nos dados tokenizados e cria arquivos com a codificação BPE"""
    with open(bpe_codes_file, "r", encoding="utf-8") as f_codes:
        bpe = BPE(f_codes)

    with open(src_file, "r", encoding="utf-8") as f_in, open(f"{output_dir}/train.src.bpe", "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(bpe.process_line(line.strip()) + "\n")

    with open(tgt_file, "r", encoding="utf-8") as f_in, open(f"{output_dir}/train.tgt.bpe", "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(bpe.process_line(line.strip()) + "\n")

    print("BPE aplicado com sucesso nos dados de treinamento!")

# Criar arquivo de configuração def create_config():
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
    
    preprocess_csv(CSV_TRAIN_PATH, output_dir)
    tokenize_file(f"{output_dir}/train.src", f"{output_dir}/train.src.tok", "en")
    tokenize_file(f"{output_dir}/train.tgt", f"{output_dir}/train.tgt.tok", "fr")
    
    learn_bpe_on_data(f"{output_dir}/train.src.tok", f"{output_dir}/train.tgt.tok", "bpe.codes")
    apply_bpe_on_data(f"{output_dir}/train.src.tok", f"{output_dir}/train.tgt.tok", "bpe.codes", output_dir)
    
    create_config()
    build_vocab()
    train_model()

if __name__ == "__main__":
    run()
