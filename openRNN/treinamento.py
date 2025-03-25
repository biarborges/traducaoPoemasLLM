import os
import requests
import zipfile
from tqdm import tqdm
import sacremoses
from subword_nmt import apply_bpe, learn_bpe

# ============== CONFIGURAÇÃO (EDITAR AQUI!) ==============
LINK_DOWNLOAD = "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/fr-pt.txt.zip"  # Substitua pelo link do dataset
SOURCE_LANG = "fr"  # Código do idioma de origem (ex: "en", "es", "de")
TARGET_LANG = "en"  # Código do idioma alvo
# ========================================================

# Configurações automáticas
DATA_DIR = f"opus_data_{SOURCE_LANG}_{TARGET_LANG}"
MODEL_DIR = f"models_{SOURCE_LANG}_{TARGET_LANG}"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def download_data():
    """Baixa os dados do link especificado"""
    try:
        print(f"Baixando {SOURCE_LANG}-{TARGET_LANG}...")
        local_path = os.path.join(DATA_DIR, "dataset.zip")
        
        # Download com barra de progresso
        response = requests.get(LINK_DOWNLOAD, stream=True)
        response.raise_for_status()
        
        with open(local_path, "wb") as f, tqdm(
            unit="B", unit_scale=True, unit_divisor=1024,
            total=int(response.headers.get('content-length', 0))
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        
        # Extração
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        os.remove(local_path)
        return True
    
    except Exception as e:
        print(f"Erro no download: {e}")
        return False

def preprocess():
    """Pré-processamento genérico para qualquer par de idiomas"""
    try:
        base_name = os.path.join(DATA_DIR, os.path.basename(LINK_DOWNLOAD).replace(".zip", ""))
        
        # Tokenização
        def tokenize_file(input_file, output_file, lang):
            tokenizer = sacremoses.MosesTokenizer(lang=lang)
            with open(input_file, "r", encoding="utf-8") as fin, \
                 open(output_file, "w", encoding="utf-8") as fout:
                for line in tqdm(fin, desc=f"Tokenizando {lang}"):
                    fout.write(tokenizer.tokenize(line.strip(), return_str=True) + "\n")
        
        tokenize_file(f"{base_name}.{SOURCE_LANG}", f"{base_name}.{SOURCE_LANG}.tok", SOURCE_LANG)
        tokenize_file(f"{base_name}.{TARGET_LANG}", f"{base_name}.{TARGET_LANG}.tok", TARGET_LANG)
        
        # BPE
        print("Aprendendo BPE...")
        with open(f"{base_name}.{SOURCE_LANG}.tok", "r") as src, \
             open(f"{base_name}.{TARGET_LANG}.tok", "r") as tgt, \
             open(os.path.join(DATA_DIR, "bpe.codes"), "w") as out:
            
            learn_bpe.learn_bpe(
                [src, tgt],
                out,
                num_symbols=10000,
                min_frequency=2
            )
        
        return True
    
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return False

def train():
    """Treinamento universal"""
    base_name = os.path.join(DATA_DIR, os.path.basename(LINK_DOWNLOAD).replace(".zip", ""))
    
    config = f"""
    data:
        corpus_1:
            path_src: {base_name}.{SOURCE_LANG}.tok
            path_tgt: {base_name}.{TARGET_LANG}.tok
        valid:
            path_src: {base_name}.valid.{SOURCE_LANG}.tok
            path_tgt: {base_name}.valid.{TARGET_LANG}.tok
    save_model: {MODEL_DIR}/model_{SOURCE_LANG}_{TARGET_LANG}
    src_vocab: {DATA_DIR}/vocab.{SOURCE_LANG}
    tgt_vocab: {DATA_DIR}/vocab.{TARGET_LANG}
    encoder_type: lstm
    decoder_type: lstm
    rnn_size: 512
    batch_size: 64
    train_steps: 50000
    """
    
    with open("config.yml", "w") as f:
        f.write(config)
    
    os.system(f"onmt_build_vocab -config config.yml -n_sample 100000")
    os.system(f"onmt_train -config config.yml")

if __name__ == "__main__":
    if download_data() and preprocess():
        train()