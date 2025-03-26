import pandas as pd
from subword_nmt import learn_bpe
import sacremoses
from tqdm import tqdm

# Caminhos dos arquivos CSV
CSV_TRAIN_PATH = "../poemas/poemas300/train/ingles_frances_train.csv"  # Arquivo de treinamento
CSV_VALID_PATH = "../poemas/poemas300/validation/ingles_frances_validation.csv"  # Arquivo de validação

# Função para pré-processamento
def preprocess_csv(csv_path):
    """Processa o CSV e cria arquivos de treinamento e validação em formato .src e .tgt"""
    df = pd.read_csv(csv_path)
    
    src_lines = []
    tgt_lines = []
    
    for _, row in df.iterrows():
        src_lines.append(row['original_poem'])
        tgt_lines.append(row['translated_poem'])

    # Salvando como arquivos .src e .tgt
    with open("train.src", "w", encoding="utf-8") as f:
        for line in src_lines:
            f.write(line + "\n")
            
    with open("train.tgt", "w", encoding="utf-8") as f:
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

    with open("bpe.codes", "w", encoding="utf-8") as f_bpe:
        learn_bpe.learn_bpe([src_lines, tgt_lines], f_bpe, num_symbols=10000, min_frequency=2)
        
    print("Aprendizado BPE concluído!")

# Função principal de execução
def run():
    """Executa todo o pipeline de pré-processamento e aprendizado de BPE"""
    preprocess_csv(CSV_TRAIN_PATH)
    tokenize_file("train.src", "train.src.tok", "en")
    tokenize_file("train.tgt", "train.tgt.tok", "fr")
    
    # Aplica o BPE
    learn_bpe_on_data("train.src.tok", "train.tgt.tok")

# Execução
if __name__ == "__main__":
    run()
