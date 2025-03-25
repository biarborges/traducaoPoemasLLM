import os
import requests
import zipfile
from tqdm import tqdm
import sacremoses
from subword_nmt import apply_bpe, learn_bpe
import time
import sys

# Configurations
PAIRS = [
    ("fr", "en"), ("fr", "pt"), 
    ("en", "fr"), ("en", "pt"), 
    ("pt", "fr"), ("pt", "en")
]
OPUS_DATASET = "OpenSubtitles"  # Try also "TED2020", "ParaCrawl" if some pairs fail
DATA_DIR = "opus_data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Download OPUS data with robust error handling ---
def download_opus(slang, tlang, max_retries=3):
    url = f"https://opus.nlpl.eu/download.php?f={OPUS_DATASET}/v1/moses/{slang}-{tlang}.txt.zip"
    zip_path = os.path.join(DATA_DIR, f"{slang}-{tlang}.zip")
    
    # Skip if already downloaded
    if os.path.exists(zip_path.replace('.zip', '')):
        print(f"Data for {slang}-{tlang} already exists. Skipping download.")
        return True
        
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Verify content type
            if 'application/zip' not in response.headers.get('Content-Type', ''):
                raise ValueError("Server didn't return a ZIP file")
                
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc=f"Downloading {slang}-{tlang}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            
            # Verify ZIP integrity
            with zipfile.ZipFile(zip_path) as zip_ref:
                if zip_ref.testzip():
                    raise zipfile.BadZipFile("Corrupt ZIP file")
                zip_ref.extractall(DATA_DIR)
            
            os.remove(zip_path)
            return True
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if attempt < max_retries - 1:
                time.sleep(5)  # Wait before retrying
                
    print(f"Failed to download {slang}-{tlang} after {max_retries} attempts")
    return False

# --- 2. Tokenization and BPE processing ---
def preprocess_data(slang, tlang):
    try:
        # Initialize tokenizers
        tokenizer_src = sacremoses.MosesTokenizer(lang=slang)
        tokenizer_tgt = sacremoses.MosesTokenizer(lang=tlang)
        
        # Input/output files
        base_filename = os.path.join(DATA_DIR, f"{OPUS_DATASET}.{slang}-{tlang}")
        src_file = f"{base_filename}.{slang}"
        tgt_file = f"{base_filename}.{tlang}"
        
        if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
            raise FileNotFoundError("Source files not found")
            
        # Tokenize
        def tokenize_file(input_file, output_file, tokenizer):
            with open(input_file, 'r', encoding='utf-8') as fin, \
                 open(output_file, 'w', encoding='utf-8') as fout:
                for line in tqdm(fin, desc=f"Tokenizing {input_file}"):
                    fout.write(tokenizer.tokenize(line.strip(), return_str=True) + '\n')
        
        src_tok = f"{base_filename}.{slang}.tok"
        tgt_tok = f"{base_filename}.{tlang}.tok"
        tokenize_file(src_file, src_tok, tokenizer_src)
        tokenize_file(tgt_file, tgt_tok, tokenizer_tgt)
        
        # Learn BPE
        bpe_code = f"{base_filename}.bpe.code"
        with open(src_tok, 'r', encoding='utf-8') as src, \
             open(tgt_tok, 'r', encoding='utf-8') as tgt, \
             open(bpe_code, 'w', encoding='utf-8') as out:
            
            # Combine both files for joint BPE
            learn_bpe.learn_bpe(
                [src, tgt],
                out,
                num_symbols=10000,
                min_frequency=2,
                verbose=False
            )
        
        # Apply BPE
        def apply_bpe_file(input_file, output_file, bpe_codes):
            with open(bpe_codes, 'r', encoding='utf-8') as codes:
                bpe = apply_bpe.BPE(codes)
                with open(input_file, 'r', encoding='utf-8') as fin, \
                     open(output_file, 'w', encoding='utf-8') as fout:
                    for line in tqdm(fin, desc=f"Applying BPE to {input_file}"):
                        fout.write(bpe.process_line(line.strip()) + '\n')
        
        src_bpe = f"{src_tok}.bpe"
        tgt_bpe = f"{tgt_tok}.bpe"
        apply_bpe_file(src_tok, src_bpe, bpe_code)
        apply_bpe_file(tgt_tok, tgt_bpe, bpe_code)
        
        return True
        
    except Exception as e:
        print(f"Error preprocessing {slang}-{tlang}: {str(e)}")
        return False

# --- 3. Train RNN model ---
def train_model(slang, tlang):
    try:
        base_filename = os.path.join(DATA_DIR, f"{OPUS_DATASET}.{slang}-{tlang}")
        config = f"""
        data:
            corpus_1:
                path_src: {base_filename}.{slang}.tok.bpe
                path_tgt: {base_filename}.{tlang}.tok.bpe
            valid:
                path_src: {base_filename}.valid.{slang}.tok.bpe
                path_tgt: {base_filename}.valid.{tlang}.tok.bpe
        save_model: {MODEL_DIR}/model_{slang}_{tlang}
        src_vocab: {base_filename}.vocab.{slang}
        tgt_vocab: {base_filename}.vocab.{tlang}
        encoder_type: lstm
        decoder_type: lstm
        rnn_size: 512
        batch_size: 64
        train_steps: 100000
        dropout: 0.3
        learning_rate: 0.001
        early_stopping: 5
        """
        
        config_file = f"config_{slang}_{tlang}.yml"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config)
        
        # Build vocabulary
        vocab_cmd = f"onmt_build_vocab -config {config_file} -n_sample 100000"
        if os.system(vocab_cmd) != 0:
            raise RuntimeError("Vocabulary building failed")
        
        # Train model
        train_cmd = f"onmt_train -config {config_file}"
        if os.system(train_cmd) != 0:
            raise RuntimeError("Training failed")
            
        return True
        
    except Exception as e:
        print(f"Error training {slang}-{tlang} model: {str(e)}")
        return False

# --- Main execution ---
if __name__ == "__main__":
    successful_pairs = []
    
    for slang, tlang in PAIRS:
        print(f"\n=== Processing {slang}-{tlang} ===")
        
        if not download_opus(slang, tlang):
            continue
            
        if not preprocess_data(slang, tlang):
            continue
            
        if train_model(slang, tlang):
            successful_pairs.append((slang, tlang))
    
    print("\n=== Summary ===")
    print(f"Successfully processed {len(successful_pairs)}/{len(PAIRS)} pairs:")
    for pair in successful_pairs:
        print(f"- {pair[0]}-{pair[1]}")