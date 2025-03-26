import pandas as pd
import sacremoses
from subword_nmt import apply_bpe, learn_bpe
from tqdm import tqdm
import os

# ============= CONFIGURAÇÃO =============

CSV_TRAIN_PATH = "../poemas/poemas300/test/frances_ingles_train.csv"  # Arquivo de treinamento
CSV_VALID_PATH = "../poemas/poemas300/test/frances_ingles_validation.csv"  # Arquivo de validação

DATA_DIR = "opus_data_en_fr"
MODEL_DIR = "models_en_fr"

# ========================================

class TranslationPreprocess:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_dir = MODEL_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.tokenizers = {
            "en": sacremoses.MosesTokenizer(lang="en"),
            "fr": sacremoses.MosesTokenizer(lang="fr")
        }

    def create_text_files(self, csv_path, src_file, tgt_file):
        """Cria arquivos de texto a partir do CSV"""
        df = pd.read_csv(csv_path)
        with open(src_file, "w", encoding="utf-8") as src_f, open(tgt_file, "w", encoding="utf-8") as tgt_f:
            for _, row in df.iterrows():
                src_f.write(row["original_poem"] + "\n")
                tgt_f.write(row["translated_poem"] + "\n")
        print(f"Arquivos {src_file} e {tgt_file} criados com sucesso!")

    def tokenize(self, src_file, tgt_file):
        """Realiza a tokenização dos arquivos de origem e destino"""
        with open(src_file, "r", encoding="utf-8") as fin, open(src_file + ".tok", "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Tokenizando {src_file}"):
                fout.write(self.tokenizers["en"].tokenize(line.strip(), return_str=True) + "\n")
        
        with open(tgt_file, "r", encoding="utf-8") as fin, open(tgt_file + ".tok", "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Tokenizando {tgt_file}"):
                fout.write(self.tokenizers["fr"].tokenize(line.strip(), return_str=True) + "\n")
        print(f"Arquivos {src_file}.tok e {tgt_file}.tok tokenizados com sucesso!")

    def learn_bpe(self, src_file, tgt_file):
        """Aplica o BPE nos arquivos tokenizados"""
        src_lines = []
        tgt_lines = []

        with open(src_file, "r", encoding="utf-8") as f:
            src_lines = [line.strip() for line in f]
        
        with open(tgt_file, "r", encoding="utf-8") as f:
            tgt_lines = [line.strip() for line in f]
        
        bpe_path = os.path.join(self.data_dir, "bpe.codes")
        if os.path.exists(bpe_path):
            print("Arquivo BPE já existe. Pulando aprendizado BPE.")
        else:
            print("Aprendendo BPE...")
            with open(bpe_path, "w", encoding="utf-8") as f_bpe:
                learn_bpe.learn_bpe(
                    [src_lines, tgt_lines],
                    f_bpe,
                    num_symbols=10000,
                    min_frequency=2
                )
            print("BPE aprendido com sucesso!")

    def apply_bpe(self, src_file, tgt_file):
        """Aplica o BPE nos arquivos de origem e destino"""
        bpe_codes = os.path.join(self.data_dir, "bpe.codes")

        # Usando a função correta para aplicar o BPE
        from subword_nmt.apply_bpe import BPE
        
        with open(bpe_codes, 'r', encoding='utf-8') as codes:
            bpe = BPE(codes)
            
            # Aplicando o BPE nos arquivos de origem e destino
            with open(src_file, 'r', encoding='utf-8') as src_in, open(src_file + ".bpe", 'w', encoding='utf-8') as src_out:
                for line in tqdm(src_in, desc=f"Aplicando BPE em {src_file}"):
                    src_out.write(bpe.process_line(line.strip()) + "\n")
            
            with open(tgt_file, 'r', encoding='utf-8') as tgt_in, open(tgt_file + ".bpe", 'w', encoding='utf-8') as tgt_out:
                for line in tqdm(tgt_in, desc=f"Aplicando BPE em {tgt_file}"):
                    tgt_out.write(bpe.process_line(line.strip()) + "\n")
        
        print(f"BPE aplicado nos arquivos {src_file} e {tgt_file}.")

    def run(self):
        """Executa todo o processo"""
        # Criar os arquivos de treinamento e validação
        self.create_text_files(CSV_TRAIN_PATH, "train.src", "train.tgt")
        self.create_text_files(CSV_VALID_PATH, "valid.src", "valid.tgt")
        
        # Tokenizar os arquivos de treinamento e validação
        self.tokenize("train.src", "train.tgt")
        self.tokenize("valid.src", "valid.tgt")
        
        # Aprender o BPE
        self.learn_bpe("train.src.tok", "train.tgt.tok")
        
        # Aplicar o BPE nos arquivos de treinamento e validação
        self.apply_bpe("train.src.tok", "train.tgt.tok")
        self.apply_bpe("valid.src.tok", "valid.tgt.tok")

if __name__ == "__main__":
    preprocess = TranslationPreprocess()
    preprocess.run()
