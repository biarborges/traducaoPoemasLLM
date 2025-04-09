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
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Arquivo do tokenizer não encontrado: {tokenizer_path}")
        
        # Carrega o modelo
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        # Carrega o tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
    
    def translate_poem(self, text, src_lang='fra_Latn', tgt_lang='eng_Latn'):
        """Traduz um poema"""
        try:
            tagged_text = f"{src_lang} {text} {tgt_lang}"
            tokens = self.tokenizer.EncodeAsPieces(tagged_text)
            token_ids = [self.tokenizer.PieceToId(token) for token in tokens]
            
            input_tensor = torch.tensor([token_ids])
            
            with torch.no_grad():
                output = self.model(input_tensor)
                translated_ids = output[0].argmax(dim=-1).tolist()
            
            translated_tokens = [self.tokenizer.IdToPiece(id) for id in translated_ids]
            return ''.join(translated_tokens).replace('▁', ' ').strip()
        
        except Exception as e:
            print(f"Erro na tradução: {e}")
            return ""

def process_csv(input_file, output_file, model_path, tokenizer_path, batch_size=1):
    """Processa o arquivo CSV completo"""
    try:
        # Carrega os dados
        df = pd.read_csv(input_file)
        
        if 'original_poem' not in df.columns:
            raise ValueError("O arquivo CSV deve conter a coluna 'original_poem'")
        
        # Inicializa o tradutor
        translator = PoemTranslator(model_path, tokenizer_path)
        
        # Tradução com barra de progresso
        translations = []
        for i in tqdm(range(len(df)), desc="Traduzindo poemas"):
            poem = df['original_poem'].iloc[i]
            try:
                translated = translator.translate_poem(poem) if pd.notna(poem) else ""
                translations.append(translated)
            except Exception as e:
                print(f"Erro no poema {i}: {e}")
                translations.append("")
            
            # Pausa para evitar sobrecarga
            if (i+1) % 10 == 0:
                time.sleep(0.1)
        
        # Adiciona resultados e salva
        df['translated_by_TA'] = translations
        df.to_csv(output_file, index=False)
        print(f"Tradução concluída! Salvo em {output_file}")
    
    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise

if __name__ == "__main__":
    # Configurações
    CONFIG = {
        "input_file": "../poemas/frances_ingles_poems_teste.csv",
        "output_file": "../poemas/OpenNMT/frances_ingles_poems_OpenNMT.csv",
        "model_path": "nllb-200-1.3B-onmt.pt",
        "tokenizer_path": "flores200_sacrebleu_tokenizer_spm.model",
        "batch_size": 1
    }
    
    # Verificação de arquivos
    missing_files = [k for k, v in CONFIG.items() 
                   if k.endswith('_path') and not os.path.exists(v)]
    
    if missing_files:
        print(f"Erro: Arquivos não encontrados - {', '.join(missing_files)}")
    else:
        process_csv(**CONFIG)