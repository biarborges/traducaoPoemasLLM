import os
from subword_nmt import apply_bpe
import sacremoses
from onmt.translate import Translator
from onmt.models import load_model

# Configurações
DATA_DIR = "opus_data_fr_pt"
MODEL_PATH = "models_fr_pt/model_fr_pt_step_50000.pt"
BPE_CODES = os.path.join(DATA_DIR, "bpe.codes")

# --- 1. Pré-processamento do texto de entrada ---
def preprocess(text, lang):
    # Tokenização
    tokenizer = sacremoses.MosesTokenizer(lang=lang)
    tokens = tokenizer.tokenize(text.strip(), return_str=True)
    
    # Aplicar BPE
    with open(BPE_CODES, "r", encoding="utf-8") as codes:
        bpe = apply_bpe.BPE(codes)
        return bpe.process_line(tokens)

# --- 2. Carregar o modelo ---
def load_translator():
    return Translator.from_checkpoint(
        MODEL_PATH,
        batch_size=1,
        beam_size=5
    )

# --- 3. Tradução ---
def translate(text):
    # Pré-processar (fr → pt)
    processed_text = preprocess(text, "fr")
    
    # Traduzir
    translator = load_translator()
    output = translator.translate([processed_text])
    
    # Pós-processamento (remover BPE)
    translated = output[0].replace("@@ ", "")
    return translated

# --- Interface simples ---
if __name__ == "__main__":
    while True:
        text = input("\nDigite o texto em francês (ou 'sair'): ")
        if text.lower() == "sair":
            break
        print("\nTradução:", translate(text))