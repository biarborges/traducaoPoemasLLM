import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import torch
from opennmt.models import NMTModel
from opennmt.utils.misc import load_checkpoint
import time
import os

class PoemTranslator:
    def __init__(self, model_path, tokenizer_path):
        """Inicializa com modelo .pt e tokenizer"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Arquivo do tokenizer não encontrado: {tokenizer_path}")
        
        # Carrega o modelo OpenNMT
        self.model = NMTModel.load(model_path)
        self.model.eval()  # Modo de avaliação
        
        # Carrega tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
    
    def translate_poem(self, text, src_lang='fra_Latn', tgt_lang='eng_Latn'):
        """Traduz um texto usando o modelo .pt"""
        try:
            # Preprocessa o texto
            tagged_text = f"{src_lang} {text} {tgt_lang}"
            tokens = self.tokenizer.EncodeAsPieces(tagged_text)
            
            # Converte para tensores
            src = torch.tensor([self.tokenizer.PieceToId(p) for p in tokens]).unsqueeze(0)
            
            # Faz a tradução
            with torch.no_grad():
                outputs = self.model(src)
                translated_ids = outputs[0].argmax(dim=-1)
            
            # Decodifica
            translated_tokens = [self.tokenizer.IdToPiece(int(i)) for i in translated_ids[0]]
            return ''.join(translated_tokens).replace('▁', ' ')
        
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return None

# ... (o resto do código da função process_csv permanece igual)

if __name__ == "__main__":
    CONFIG = {
        "input_file": "../poemas/frances_ingles_poems_teste.csv",
        "output_file": "../poemas/OpenNMT/frances_ingles_poems_OpenNMT.csv",
        "model_path": "nllb-200-1.3B-onmt.pt",  # Arquivo .pt original
        "tokenizer_path": "flores200_sacrebleu_tokenizer_spm.model",
        "batch_size": 1  # Reduzido para 1 por limitações de memória
    }
    
    # Verifica se os arquivos existem
    if not all(os.path.exists(p) for p in [CONFIG["model_path"], CONFIG["tokenizer_path"]]):
        print("Erro: Verifique se os caminhos dos arquivos estão corretos")
    else:
        process_csv(**CONFIG)