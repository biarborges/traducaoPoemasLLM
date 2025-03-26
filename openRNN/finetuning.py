import os
import pandas as pd
from subword_nmt import learn_bpe, apply_bpe
import sacremoses
from tqdm import tqdm

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
def learn_bpe_on_data(src_file, tgt_file):
    """Aplica o BPE aos dados"""
    src_lines = []
    tgt_lines = []
    
    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f.readlines()]
        
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_lines = [line.strip() for line in f.readlines()]

    # Aplicando o BPE separadamente para src e tgt
    with open("bpe.codes", "w", encoding="utf-8") as f_bpe:
        learn_bpe.learn_bpe(src_lines + tgt_lines, f_bpe, num_symbols=10000, min_frequency=2)
        
    print("Aprendizado BPE concluído!")

# Função para aplicar BPE nos arquivos de dados
def apply_bpe_on_data(src_file, tgt_file, bpe_codes_file, output_dir):
    """Aplica o BPE nos dados tokenizados e cria arquivos com a codificação BPE"""
    with open(src_file, "r", encoding="utf-8") as f_in, open(f"{output_dir}/train.src.bpe", "w", encoding="utf-8") as f_out:
        apply_bpe.apply_bpe(f_in, f_out, bpe_codes_file)
    
    with open(tgt_file, "r", encoding="utf-8") as f_in, open(f"{output_dir}/train.tgt.bpe", "w", encoding="utf-8") as f_out:
        apply_bpe.apply_bpe(f_in, f_out, bpe_codes_file)
    
    print("BPE aplicado com sucesso nos dados de treinamento!")

# Função principal de execução
def run():
    """Executa todo o pipeline de pré-processamento, tokenização, aprendizado de BPE e aplicação de BPE"""
    output_dir = "output_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Passo 1: Pré-processamento (converter CSV para arquivos .src e .tgt)
    preprocess_csv(CSV_TRAIN_PATH, output_dir)
    
    # Passo 2: Tokenização dos arquivos .src e .tgt
    tokenize_file(f"{output_dir}/train.src", f"{output_dir}/train.src.tok", "en")
    tokenize_file(f"{output_dir}/train.tgt", f"{output_dir}/train.tgt.tok", "fr")
    
    # Passo 3: Aprender o BPE
    learn_bpe_on_data(f"{output_dir}/train.src.tok", f"{output_dir}/train.tgt.tok")
    
    # Passo 4: Aplicar o BPE nos dados tokenizados
    apply_bpe_on_data(f"{output_dir}/train.src.tok", f"{output_dir}/train.tgt.tok", "bpe.codes", output_dir)

# Execução
if __name__ == "__main__":
    run()
