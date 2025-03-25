import os
import requests
import zipfile
import gzip
import shutil
from tqdm import tqdm
import sacremoses
from subword_nmt import apply_bpe, learn_bpe
import time

# Configurações específicas para fr-pt
DATA_DIR = "opus_data_fr_pt"
MODEL_DIR = "models_fr_pt"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Download dos dados ---
def download_fr_pt():
    url = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/fr-pt_BR.txt.zip"
    zip_path = os.path.join(DATA_DIR, "fr-pt.zip")
    
    try:
        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Baixando fr-pt",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        # Extração
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        print("Download e extração concluídos!")
        return True
    
    except Exception as e:
        print(f"Erro no download: {e}")
        return False

# --- 2. Pré-processamento ---
def preprocess_fr_pt():
    try:
        # Caminhos dos arquivos
        fr_file = os.path.join(DATA_DIR, "OpenSubtitles.fr-pt_BR.fr")
        pt_file = os.path.join(DATA_DIR, "OpenSubtitles.fr-pt_BR.pt_BR")
        
        # Tokenização
        tokenizer_fr = sacremoses.MosesTokenizer(lang="fr")
        tokenizer_pt = sacremoses.MosesTokenizer(lang="pt")
        
        def tokenize(input_file, output_file, tokenizer):
            with open(input_file, "r", encoding="utf-8") as fin, \
                 open(output_file, "w", encoding="utf-8") as fout:
                for line in tqdm(fin, desc=f"Tokenizando {input_file}"):
                    fout.write(tokenizer.tokenize(line.strip(), return_str=True) + "\n")
        
        tokenize(fr_file, fr_file + ".tok", tokenizer_fr)
        tokenize(pt_file, pt_file + ".tok", tokenizer_pt)
        
        # BPE
        with open(fr_file + ".tok", "r", encoding="utf-8") as fr, \
             open(pt_file + ".tok", "r", encoding="utf-8") as pt, \
             open(os.path.join(DATA_DIR, "bpe.codes"), "w", encoding="utf-8") as bpe_out:
            
            learn_bpe.learn_bpe(
                [fr, pt],
                bpe_out,
                num_symbols=10000,
                min_frequency=2
            )
        
        print("Pré-processamento concluído!")
        return True
    
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return False

# --- 3. Treinamento ---
def train_fr_pt():
    config = f"""
    data:
        corpus_1:
            path_src: {DATA_DIR}/OpenSubtitles.fr-pt_BR.fr.tok
            path_tgt: {DATA_DIR}/OpenSubtitles.fr-pt_BR.pt_BR.tok
        valid:
            path_src: {DATA_DIR}/valid.fr.tok
            path_tgt: {DATA_DIR}/valid.pt.tok
    save_model: {MODEL_DIR}/model_fr_pt
    src_vocab: {DATA_DIR}/vocab.fr
    tgt_vocab: {DATA_DIR}/vocab.pt
    encoder_type: lstm
    decoder_type: lstm
    rnn_size: 512
    batch_size: 64
    dropout: 0.3
    train_steps: 50000
    """
    
    with open("config_fr_pt.yml", "w", encoding="utf-8") as f:
        f.write(config)
    
    # Comandos de treinamento
    os.system(f"onmt_build_vocab -config config_fr_pt.yml -n_sample 100000")
    os.system(f"onmt_train -config config_fr_pt.yml")

# --- Execução principal ---
if __name__ == "__main__":
    start_time = time.time()
    if download_fr_pt() and preprocess_fr_pt():
        train_fr_pt()
    end_time = time.time()
    print(f"\nTempo total: {end_time - start_time:.2f} segundos.")