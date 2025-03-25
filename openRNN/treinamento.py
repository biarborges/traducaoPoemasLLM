import os
import requests
import zipfile
from tqdm import tqdm
import sacremoses
from subword_nmt import apply_bpe, learn_bpe

# ============== APENAS TROCAR ESTES 3 VALORES ==============
DATASET_NAME = "KDE4"         # Nome do dataset (OPUS)
LANG_PAIR = ("en", "fr")      # Par de idiomas (origem, alvo)
BASE_URL = "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-fr.txt.zip"
# ===========================================================

# Configurações automáticas
src, tgt = LANG_PAIR
DATA_DIR = f"opus_data_{src}_{tgt}"
MODEL_DIR = f"models_{src}_{tgt}"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def run_pipeline():
    """Executa todo o pipeline com as configurações definidas"""
    # 1. Download
    url = BASE_URL.format(dataset=DATASET_NAME, src=src, tgt=tgt)
    zip_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.zip")
    
    print(f"Baixando {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(zip_path, "wb") as f, tqdm(
        unit="B", unit_scale=True,
        total=int(response.headers.get('content-length', 0))
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            bar.update(len(chunk))
            f.write(chunk)
    
    # 2. Extração
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)
    
    # 3. Pré-processamento
    base_file = os.path.join(DATA_DIR, f"{DATASET_NAME}.{src}-{tgt}")
    tokenizers = {
        src: sacremoses.MosesTokenizer(lang=src),
        tgt: sacremoses.MosesTokenizer(lang=tgt)
    }
    
    for lang in LANG_PAIR:
        with open(f"{base_file}.{lang}", "r", encoding="utf-8") as fin, \
             open(f"{base_file}.{lang}.tok", "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Tokenizando {lang}"):
                fout.write(tokenizers[lang].tokenize(line.strip(), return_str=True) + "\n")
    
    # 4. BPE
    with open(f"{base_file}.{src}.tok", "r") as f_src, \
         open(f"{base_file}.{tgt}.tok", "r") as f_tgt, \
         open(os.path.join(DATA_DIR, "bpe.codes"), "w") as f_bpe:
        
        learn_bpe.learn_bpe(
            [f_src, f_tgt],
            f_bpe,
            num_symbols=10000,
            min_frequency=2
        )
    
    # 5. Treinamento
    config = f"""
    data:
        corpus_1:
            path_src: {base_file}.{src}.tok
            path_tgt: {base_file}.{tgt}.tok
        valid:
            path_src: {base_file}.valid.{src}.tok
            path_tgt: {base_file}.valid.{tgt}.tok
    save_model: {MODEL_DIR}/model_{src}_{tgt}
    src_vocab: {DATA_DIR}/vocab.{src}
    tgt_vocab: {DATA_DIR}/vocab.{tgt}
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
    run_pipeline()
    print(f"\nPipeline completo! Modelo salvo em: {MODEL_DIR}/")