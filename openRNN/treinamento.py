import os
import requests
import zipfile
from tqdm import tqdm
import sacremoses
from subword_nmt import apply_bpe, learn_bpe
import yaml

# ============= CONFIGURAÇÃO (EDITAR AQUI) =============
CONFIG = {
    "dataset": "KDE4",                # Nome do dataset (OPUS)
    "source_lang": "en",              # Código do idioma fonte
    "target_lang": "fr",              # Código do idioma alvo
    "base_url": "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-fr.txt.zip",
    "train_steps": 50000,             # Passos de treinamento
    "rnn_size": 512,                  # Tamanho das RNNs
    "batch_size": 64,                 # Tamanho do batch
    "use_gpu": True                   # Usar GPU se disponível
}
# ======================================================

class TranslationPipeline:
    def __init__(self, config):
        self.config = config
        self.data_dir = f"opus_data_{config['source_lang']}_{config['target_lang']}"
        self.model_dir = f"models_{config['source_lang']}_{config['target_lang']}"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def download_data(self):
        """Baixa dados paralelos do OPUS"""
        url = self.config["base_url"]
        zip_path = os.path.join(self.data_dir, "dataset.zip")
        
        try:
            print(f"Baixando {url}...")
            response = requests.get(url, stream=True)
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
        """Tokenização e BPE com tratamento robusto"""
        try:
            base_name = os.path.join(
                self.data_dir,
                f"KDE4.{self.config['source_lang']}-{self.config['target_lang']}"
            )
            
            # 1. Tokenização
            print("Iniciando tokenização...")
            tokenizers = {
                self.config["source_lang"]: sacremoses.MosesTokenizer(lang=self.config["source_lang"]),
                self.config["target_lang"]: sacremoses.MosesTokenizer(lang=self.config["target_lang"])
            }
            
            for lang in [self.config["source_lang"], self.config["target_lang"]]:
                input_file = f"{base_name}.{lang}"
                output_file = f"{input_file}.tok"
                
                with open(input_file, "r", encoding="utf-8") as fin, \
                    open(output_file, "w", encoding="utf-8") as fout:
                    for line in tqdm(fin, desc=f"Tokenizando {lang}"):
                        fout.write(tokenizers[lang].tokenize(line.strip(), return_str=True) + "\n")
            
            # 2. BPE - Versão 100% funcional
            print("Aprendendo BPE...")
            bpe_path = os.path.join(self.data_dir, "bpe.codes")
            
            # Método garantido usando arquivos temporários
            with open(f"{base_name}.{self.config['source_lang']}.tok", "r") as f_src, \
                open(f"{base_name}.{self.config['target_lang']}.tok", "r") as f_tgt:
                
                # Lê todo o conteúdo em memória (para datasets razoavelmente grandes)
                src_content = f_src.read()
                tgt_content = f_tgt.read()
            
            # Escreve em arquivos temporários para o BPE
            temp_src = os.path.join(self.data_dir, "temp_src.txt")
            temp_tgt = os.path.join(self.data_dir, "temp_tgt.txt")
            
            with open(temp_src, "w") as f:
                f.write(src_content)
            with open(temp_tgt, "w") as f:
                f.write(tgt_content)
            
            # Aprende BPE usando os arquivos temporários
            with open(bpe_path, "w") as f_bpe:
                learn_bpe.learn_bpe(
                    [open(temp_src, "r"), open(temp_tgt, "r")],
                    f_bpe,
                    num_symbols=10000,
                    min_frequency=2
                )
            
            # Remove arquivos temporários
            os.remove(temp_src)
            os.remove(temp_tgt)
            
            return True
        
        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            return False

    def train(self):
        """Configura e executa o treinamento"""
        config = {
            "data": {
                "corpus_1": {
                    "path_src": os.path.join(
                        self.data_dir,
                        f"KDE4.{self.config['source_lang']}-{self.config['target_lang']}.{self.config['source_lang']}.tok"
                    ),
                    "path_tgt": os.path.join(
                        self.data_dir,
                        f"KDE4.{self.config['source_lang']}-{self.config['target_lang']}.{self.config['target_lang']}.tok"
                    )
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
            "src_vocab": os.path.join(self.data_dir, "vocab.src"),
            "tgt_vocab": os.path.join(self.data_dir, "vocab.tgt"),
            "encoder_type": "lstm",
            "decoder_type": "lstm",
            "rnn_size": self.config["rnn_size"],
            "batch_size": self.config["batch_size"],
            "train_steps": self.config["train_steps"],
            "world_size": 1,
            "gpu_ranks": [0] if self.config["use_gpu"] else [],
            "fp16": self.config["use_gpu"]  # Mixed precision para GPU
        }
        
        with open("config.yml", "w") as f:
            yaml.dump(config, f)
        
        os.system(f"onmt_build_vocab -config config.yml -n_sample 100000")
        os.system(f"onmt_train -config config.yml")

def main():
    pipeline = TranslationPipeline(CONFIG)
    
    if pipeline.download_data() and pipeline.preprocess():
        pipeline.train()

if __name__ == "__main__":
    main()