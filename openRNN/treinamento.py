import os
import requests
import zipfile
from tqdm import tqdm
import sacremoses
import subword_nmt.apply_bpe
import subword_nmt.learn_bpe

# Configurações
PAIRS = [
    ("fr", "en"), ("fr", "pt"), 
    ("en", "fr"), ("en", "pt"), 
    ("pt", "fr"), ("pt", "en")
]
OPUS_DATASET = "OpenSubtitles"  # Dataset do OPUS (pode ser "OpenSubtitles", "ParaCrawl", etc.)
DATA_DIR = "opus_data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Baixar dados do OPUS ---
def download_opus(slang, tlang):
    url = f"https://opus.nlpl.eu/download.php?f={OPUS_DATASET}/v1/moses/{slang}-{tlang}.txt.zip"
    zip_path = os.path.join(DATA_DIR, f"{slang}-{tlang}.zip")
    
    # Download (com barra de progresso)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        desc=f"Baixando {slang}-{tlang}",
        total=total_size,
        unit="B",
        unit_scale=True,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            progress.update(len(chunk))
    
    # Extrair
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)

# --- 2. Tokenizar e aplicar BPE ---
def preprocess_data(slang, tlang):
    # Tokenizadores
    tokenizer_src = sacremoses.MosesTokenizer(lang=slang)
    tokenizer_tgt = sacremoses.MosesTokenizer(lang=tlang)
    
    # Arquivos de entrada/saída
    src_file = os.path.join(DATA_DIR, f"{OPUS_DATASET}.{slang}-{tlang}.{slang}")
    tgt_file = os.path.join(DATA_DIR, f"{OPUS_DATASET}.{slang}-{tlang}.{tlang}")
    src_tok = os.path.join(DATA_DIR, f"train.{slang}.tok")
    tgt_tok = os.path.join(DATA_DIR, f"train.{tlang}.tok")
    
    # Tokenização
    with open(src_file, "r") as f_src, open(src_tok, "w") as f_tok_src:
        for line in tqdm(f_src, desc=f"Tokenizando {slang}"):
            f_tok_src.write(tokenizer_src.tokenize(line.strip(), return_str=True) + "\n")
    
    with open(tgt_file, "r") as f_tgt, open(tgt_tok, "w") as f_tok_tgt:
        for line in tqdm(f_tgt, desc=f"Tokenizando {tlang}"):
            f_tok_tgt.write(tokenizer_tgt.tokenize(line.strip(), return_str=True) + "\n")
    
    # Aprender BPE
    bpe_code = os.path.join(DATA_DIR, f"bpe.{slang}-{tlang}.code")
    subword_nmt.learn_bpe.learn_bpe(
        open(src_tok, "r"), open(bpe_code, "w"), num_symbols=10000
    )
    
    # Aplicar BPE
    bpe = subword_nmt.apply_bpe.BPE(open(bpe_code, "r"))
    with open(src_tok, "r") as f_in, open(f"{src_tok}.bpe", "w") as f_out:
        for line in tqdm(f_in, desc=f"Aplicando BPE em {slang}"):
            f_out.write(bpe.process_line(line.strip()) + "\n")
    
    with open(tgt_tok, "r") as f_in, open(f"{tgt_tok}.bpe", "w") as f_out:
        for line in tqdm(f_in, desc=f"Aplicando BPE em {tlang}"):
            f_out.write(bpe.process_line(line.strip()) + "\n")

# --- 3. Treinar modelo RNN (LSTM) ---
def train_model(slang, tlang):
    config = f"""
    data:
        corpus_1:
            path_src: {DATA_DIR}/train.{slang}.tok.bpe
            path_tgt: {DATA_DIR}/train.{tlang}.tok.bpe
        valid:
            path_src: {DATA_DIR}/valid.{slang}.tok.bpe
            path_tgt: {DATA_DIR}/valid.{tlang}.tok.bpe
    save_model: {MODEL_DIR}/model_{slang}_{tlang}
    src_vocab: {DATA_DIR}/vocab.{slang}
    tgt_vocab: {DATA_DIR}/vocab.{tlang}
    encoder_type: lstm
    decoder_type: lstm
    rnn_size: 512
    batch_size: 64
    train_steps: 100000
    """
    
    with open(f"config_{slang}_{tlang}.yml", "w") as f:
        f.write(config)
    
    # Construir vocabulário
    os.system(f"onmt_build_vocab -config config_{slang}_{tlang}.yml -n_sample 100000")
    
    # Treinar
    os.system(f"onmt_train -config config_{slang}_{tlang}.yml")

# --- Execução principal ---
if __name__ == "__main__":
    # Baixar e pré-processar dados para todos os pares
    for slang, tlang in PAIRS:
        print(f"\n=== Processando {slang}-{tlang} ===")
        download_opus(slang, tlang)
        preprocess_data(slang, tlang)
        train_model(slang, tlang)