import os
import requests
import zipfile
from tqdm import tqdm
import sacremoses
from subword_nmt import learn_bpe
import yaml
import time
import random

start_time = time.time()

# ============= CONFIGURAÇÃO ============= 
CONFIG = {
    "dataset": "TED2020",
    "source_lang": "pt",
    "target_lang": "en",
    "base_url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-pt.txt.zip",
    "train_steps": 100000,
    "rnn_size": 512,
    "batch_size": 16,
    "use_gpu": True
}
# ========================================

class TranslationPipeline:
    def __init__(self, config):
        self.config = config
        self.data_dir = f"opus_data_{config['source_lang']}_{config['target_lang']}"
        self.model_dir = f"models_{config['source_lang']}_{config['target_lang']}"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Nomes de arquivos base
        self.base_name = os.path.join(
            self.data_dir,
            f"TED2020.{self.config['source_lang']}-{self.config['target_lang']}"
        )

    def _check_files_exist(self):
        """Verifica se os arquivos necessários já existem"""
        required_files = [
            f"{self.base_name}.{self.config['source_lang']}",
            f"{self.base_name}.{self.config['target_lang']}"
        ]
        return all(os.path.exists(f) for f in required_files)

    def download_data(self):
        """Baixa dados apenas se necessário"""
        if self._check_files_exist():
            print("Arquivos já existem. Pulando download.")
            return True
            
        try:
            print(f"Baixando {self.config['base_url']}...")
            zip_path = os.path.join(self.data_dir, "dataset.zip")
            
            response = requests.get(self.config["base_url"], stream=True)
            response.raise_for_status()
            
            with open(zip_path, "wb") as f, tqdm(
                unit="B", unit_scale=True,
                total=int(response.headers.get('content-length', 0))
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            os.remove(zip_path)
            return True
            
        except Exception as e:
            print(f"Erro no download: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False

    def preprocess(self):
        """Pré-processamento com tratamento robusto"""
        try:
            # Verifica se os arquivos tokenizados já existem
            if all(os.path.exists(f"{self.base_name}.{lang}.tok") for lang in [self.config["source_lang"], self.config["target_lang"]] ):
                print("Arquivos tokenizados já existem. Pulando tokenização.")
            else:
                print("Iniciando tokenização...")
                tokenizers = {
                    self.config["source_lang"]: sacremoses.MosesTokenizer(lang=self.config["source_lang"]),
                    self.config["target_lang"]: sacremoses.MosesTokenizer(lang=self.config["target_lang"])
                }
                
                for lang in [self.config["source_lang"], self.config["target_lang"]]:
                    input_file = f"{self.base_name}.{lang}"
                    output_file = f"{input_file}.tok"
                    
                    with open(input_file, "r", encoding="utf-8") as fin, \
                         open(output_file, "w", encoding="utf-8") as fout:
                        for line in tqdm(fin, desc=f"Tokenizando {lang}"):
                            fout.write(tokenizers[lang].tokenize(line.strip(), return_str=True) + "\n")
            
            # Dividir 10% do treinamento para validação
            self.create_validation_set()
            
            # Processamento BPE
            bpe_path = os.path.join(self.data_dir, "bpe.codes")
            if os.path.exists(bpe_path):
                print("Arquivo BPE já existe. Pulando aprendizado.")
            else:
                print("Aprendendo BPE...")
                src_lines = []
                tgt_lines = []
                
                with open(f"{self.base_name}.{self.config['source_lang']}.tok", "r") as f:
                    src_lines = [line.strip() for line in f.readlines()]
                
                with open(f"{self.base_name}.{self.config['target_lang']}.tok", "r") as f:
                    tgt_lines = [line.strip() for line in f.readlines()]
                
                with open(bpe_path, "w") as f_bpe:
                    with open(f"{self.base_name}.{self.config['source_lang']}.tok", "r") as src_file, \
                        open(f"{self.base_name}.{self.config['target_lang']}.tok", "r") as tgt_file:
                        learn_bpe.learn_bpe(
                            (line.strip() for line in src_file),  # Passando iterador
                            f_bpe,
                            num_symbols=10000,
                            min_frequency=2
                        )
            
            return True
            
        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            return False
    
    def create_validation_set(self):
        """Cria 10% do conjunto de validação a partir do treinamento"""
        try:
            # Abra os arquivos tokenizados
            src_lines = []
            tgt_lines = []
            
            with open(f"{self.base_name}.{self.config['source_lang']}.tok", "r") as f_src:
                src_lines = [line.strip() for line in f_src.readlines()]
            
            with open(f"{self.base_name}.{self.config['target_lang']}.tok", "r") as f_tgt:
                tgt_lines = [line.strip() for line in f_tgt.readlines()]
            
            # Embaralha os dados e seleciona 10% para validação
            combined = list(zip(src_lines, tgt_lines))
            random.shuffle(combined)
            valid_size = int(0.1 * len(combined))  # 10% para validação
            valid_data = combined[:valid_size]
            train_data = combined[valid_size:]
            
            # Salva os arquivos de validação
            with open(os.path.join(self.data_dir, "valid.src.tok"), "w", encoding="utf-8") as f_src, \
                 open(os.path.join(self.data_dir, "valid.tgt.tok"), "w", encoding="utf-8") as f_tgt:
                for src, tgt in valid_data:
                    f_src.write(src + "\n")
                    f_tgt.write(tgt + "\n")
            
            # Salva os arquivos de treinamento
            with open(f"{self.base_name}.{self.config['source_lang']}.tok", "w", encoding="utf-8") as f_src, \
                 open(f"{self.base_name}.{self.config['target_lang']}.tok", "w", encoding="utf-8") as f_tgt:
                for src, tgt in train_data:
                    f_src.write(src + "\n")
                    f_tgt.write(tgt + "\n")
            
        except Exception as e:
            print(f"Erro ao criar o conjunto de validação: {e}")

    def train(self):
        """Configura e executa o treinamento"""
        config = {
            "data": {
                "corpus_1": {
                    "path_src": f"{self.base_name}.{self.config['source_lang']}.tok",
                    "path_tgt": f"{self.base_name}.{self.config['target_lang']}.tok"
                },
                "valid": {
                    "path_src": os.path.join(self.data_dir, "valid.src.tok"),
                    "path_tgt": os.path.join(self.data_dir, "valid.tgt.tok")
                }
            },
            "save_model": os.path.join(
                self.model_dir,
                f"model_{self.config['source_lang']}_{self.config['target_lang']}"
            ),
            "save_data": os.path.join(self.model_dir, f"train_{self.config['source_lang']}_{self.config['target_lang']}"),
            "src_vocab": os.path.join(self.data_dir, "vocab.src"),
            "tgt_vocab": os.path.join(self.data_dir, "vocab.tgt"),
            "encoder_type": "rnn",
            "decoder_type": "rnn",
            "rnn_size": self.config["rnn_size"],
            "batch_size": self.config["batch_size"],
            "train_steps": self.config["train_steps"],
            "world_size": 2,
            "gpu_ranks": [0] if self.config["use_gpu"] else [],
            "fp16": self.config["use_gpu"]
        }
        
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
        
        # Primeiro, gera os vocabulários
        os.system(f"onmt_build_vocab -config config.yaml -n_sample 100000")
        
        # Em seguida, inicia o treinamento
        os.system(f"onmt_train -config config.yaml")

def main():
    pipeline = TranslationPipeline(CONFIG)
    
    if pipeline.download_data() and pipeline.preprocess():
        pipeline.train()

if __name__ == "__main__":
    main()
    print(f"Tempo total: {time.time() - start_time:.2f} segundos")
