import os
from multiprocessing import Pool
import sacremoses
from subword_nmt import apply_bpe, learn_bpe
from onmt_train import train
from onmt.translate import Translator

# Configurações (ajuste esses paths!)
DATA_DIR = "opus_mini"
MODEL_DIR = "models_gpu"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Baixar APENAS 1M de linhas (amostra) ---
def download_sample():
    os.system(f"wget -qO- https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2024/moses/fr-pt_BR.txt.zip | head -n 1000000 > {DATA_DIR}/sample.fr-pt.zip")
    os.system(f"unzip -j {DATA_DIR}/sample.fr-pt.zip -d {DATA_DIR}")

# --- 2. Pré-processamento paralelizado (CPU) ---
def tokenize_parallel(input_file, output_file, lang):
    tokenizer = sacremoses.MosesTokenizer(lang=lang)
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        with Pool() as pool:
            for line in pool.imap(tokenizer.tokenize, f_in, chunksize=1000):
                f_out.write(" ".join(line) + "\n")

# --- 3. Treino ACELERADO (GPU) ---
def train_gpu():
    config = {
        'data': {
            'corpus_1': {
                'path_src': f"{DATA_DIR}/OpenSubtitles.fr-pt_BR.fr.tok",
                'path_tgt': f"{DATA_DIR}/OpenSubtitles.fr-pt_BR.pt_BR.tok"
            },
            'valid': {
                'path_src': f"{DATA_DIR}/valid.fr.tok",
                'path_tgt': f"{DATA_DIR}/valid.pt.tok"
            }
        },
        'world_size': 1,
        'gpu_ranks': [0],
        'save_model': f"{MODEL_DIR}/model_fr_pt",
        'encoder_type': 'lstm',
        'decoder_type': 'lstm',
        'train_steps': 20000,  # Reduzido para teste
        'fp16': True  # Ativa mixed-precision
    }
    train(config)

# --- 4. Tradução RÁPIDA (GPU) ---
def translate_gpu(text):
    translator = Translator(
        model_path=f"{MODEL_DIR}/model_fr_pt_step_20000.pt",
        gpu=0
    )
    return translator.translate([text])[0]

# --- Pipeline completo ---
if __name__ == "__main__":
    print("1. Baixando amostra...")
    download_sample()
    
    print("\n2. Tokenizando (paralelizado)...")
    tokenize_parallel(f"{DATA_DIR}/OpenSubtitles.fr-pt_BR.fr", f"{DATA_DIR}/OpenSubtitles.fr-pt_BR.fr.tok", "fr")
    tokenize_parallel(f"{DATA_DIR}/OpenSubtitles.fr-pt_BR.pt_BR", f"{DATA_DIR}/OpenSubtitles.fr-pt_BR.pt_BR.tok", "pt")
    
    print("\n3. Treinando com GPU...")
    train_gpu()
    
    print("\n4. Testando tradução:")
    print(translate_gpu("Bonjour comment allez-vous?"))