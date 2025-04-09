import pandas as pd
import sentencepiece as spm
import torch
from tqdm import tqdm
import time
import os

class PoemTranslator:
    def __init__(self, model_path, tokenizer_path):
        """Carrega o modelo e tokenizer diretamente com PyTorch"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        # Carrega o modelo diretamente com PyTorch
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Carrega o tokenizer SentencePiece
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
    
    def translate_poem(self, text, src_lang='fra_Latn', tgt_lang='eng_Latn'):
        """Tradução direta com PyTorch"""
        try:
            # Prepara o texto com tags de idioma
            tagged_text = f"{src_lang} {text} {tgt_lang}"
            
            # Tokeniza
            tokens = self.tokenizer.EncodeAsPieces(tagged_text)
            token_ids = [self.tokenizer.PieceToId(token) for token in tokens]
            
            # Converte para tensor
            input_tensor = torch.tensor([token_ids])
            
            # Faz a inferência
            with torch.no_grad():
                output = self.model(input_tensor)
                translated_ids = output[0].argmax(dim=-1).tolist()
            
            # Decodifica
            translated_tokens = [self.tokenizer.IdToPiece(id) for id in translated_ids]
            return ''.join(translated_tokens).replace('▁', ' ').strip()
        
        except Exception as e:
            print(f"Translation error: {e}")
            return ""

# ... (mantenha o restante do código igual, incluindo process_csv e main)

if __name__ == "__main__":
    # Configurações
    CONFIG = {
        "input_file": "../poemas/frances_ingles_poems_teste.csv",
        "output_file": "../poemas/OpenNMT/frances_ingles_poems_OpenNMT.csv",
        "model_path": "nllb-200-1.3B-onmt.pt",
        "tokenizer_path": "flores200_sacrebleu_tokenizer_spm.model",
        "batch_size": 1  # Mantenha 1 para modelos grandes
    }
    
    # Verificação de arquivos
    missing_files = [k for k, v in CONFIG.items() 
                   if k.endswith('_path') and not os.path.exists(v)]
    
    if missing_files:
        print(f"Error: Missing files - {', '.join(missing_files)}")
    else:
        process_csv(**CONFIG)