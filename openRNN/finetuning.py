import pandas as pd
import os
from subword_nmt import apply_bpe, learn_bpe
import sacremoses
from tqdm import tqdm

# Caminhos dos arquivos CSV
CSV_TRAIN_PATH = "../poemas/poemas300/train/ingles_frances_train.csv"  # Arquivo de treinamento
CSV_VALID_PATH = "../poemas/poemas300/validation/ingles_frances_validation.csv"  # Arquivo de validação

# Diretórios de saída
DATA_DIR = "opus_data"
MODEL_DIR = "models"

# Função para pré-processamento
def preprocess_csv(csv_path):
    """Processa o CSV e cria arquivos de treinamento e validação em formato .src e .tgt"""
    df = pd.read_csv(csv_path)
    
    # Criar arquivos de texto separados para src e tgt
    src_file = csv_path.replace(".csv", ".src")
    tgt_file = csv_path.replace(".csv", ".tgt")
    
    with open(src_file, "w", encoding="utf-8") as f_src, open(tgt_file, "w", encoding="utf-8") as f_tgt:
        for _, row in df.iterrows():
            f_src.write(row["original_poem"] + "\n")
            f_tgt.write(row["translated_poem"] + "\n")
    
    print(f"Arquivos {src_file} e {tgt_file} criados com sucesso!")

    return src_file, tgt_file

# Função para tokenização
def tokenize_file(file_path, lang):
    """Tokeniza os arquivos de entrada usando o sacremoses"""
    tokenizer = sacremoses.MosesTokenizer(lang=lang)
    tokenized_file = file_path + ".tok"
    
    with open(file_path, "r", encoding="utf-8") as f_in, open(tokenized_file, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc=f"Tokenizando {lang}"):
            f_out.write(tokenizer.tokenize(line.strip(), return_str=True) + "\n")
    
    print(f"Arquivo {tokenized_file} tokenizado com sucesso!")
    return tokenized_file

# Função para aprender e aplicar o BPE
def learn_bpe_on_data(src_file, tgt_file):
    """Aprende o BPE a partir dos arquivos de treino e validação"""
    src_lines = []
    tgt_lines = []
    
    with open(src_file, "r") as f:
        for line in f:
            src_lines.append(line.strip())  # Adicionando linha ao vetor

    with open(tgt_file, "r") as f:
        for line in f:
            tgt_lines.append(line.strip())  # Adicionando linha ao vetor

    print("Aprendendo o BPE...")

    # Aqui, estamos passando as linhas de forma correta
    with open("bpe.codes", "w") as f_bpe:
        learn_bpe.learn_bpe([src_lines, tgt_lines], f_bpe, num_symbols=10000, min_frequency=2)

    print("BPE aprendido com sucesso!")

# Função para aplicar BPE
def apply_bpe_to_files(src_file, tgt_file, bpe_codes):
    """Aplica o BPE nos arquivos de origem e destino"""
    apply_bpe.apply_bpe(src_file, src_file + ".bpe", bpe_codes)
    apply_bpe.apply_bpe(tgt_file, tgt_file + ".bpe", bpe_codes)
    print(f"BPE aplicado em {src_file} e {tgt_file}.")

# Função principal
def run():
    # Pré-processamento dos dados
    src_train_file, tgt_train_file = preprocess_csv(CSV_TRAIN_PATH)
    src_valid_file, tgt_valid_file = preprocess_csv(CSV_VALID_PATH)
    
    # Tokenização dos arquivos
    src_train_tok = tokenize_file(src_train_file, lang="en")
    tgt_train_tok = tokenize_file(tgt_train_file, lang="fr")
    src_valid_tok = tokenize_file(src_valid_file, lang="en")
    tgt_valid_tok = tokenize_file(tgt_valid_file, lang="fr")
    
    # Aprendizado do BPE
    bpe_codes = "bpe.codes"
    if not os.path.exists(bpe_codes):
        learn_bpe_on_data(src_train_tok, tgt_train_tok)
    else:
        print("Arquivo BPE já existe. Pulando aprendizado.")
    
    # Aplicando o BPE nos dados de treino e validação
    apply_bpe_to_files(src_train_tok, tgt_train_tok, bpe_codes)
    apply_bpe_to_files(src_valid_tok, tgt_valid_tok, bpe_codes)

    print("Pré-processamento concluído!")

if __name__ == "__main__":
    run()
