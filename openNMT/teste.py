import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import time
import ctranslate2

class PoemTranslator:
    def __init__(self, model_path, tokenizer_path):
        """Inicializa o tradutor com modelo e tokenizer
        
        Args:
            model_path: caminho para o modelo .pt
            tokenizer_path: caminho para o tokenizer .model
        """
        try:
            self.translator = ctranslate2.Translator(model_path)
            self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
            print("Modelo e tokenizer carregados com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo/tokenizer: {e}")
            raise

    def translate_poem(self, text, src_lang='fra_Latn', tgt_lang='eng_Latn'):
        """Traduz um texto usando o modelo NLLB
        
        Args:
            text: texto a ser traduzido
            src_lang: código do idioma de origem
            tgt_lang: código do idioma de destino
            
        Returns:
            Texto traduzido ou None em caso de erro
        """
        try:
            tagged_text = f"{src_lang} {text} {tgt_lang}"
            tokens = self.tokenizer.encode(tagged_text)
            results = self.translator.translate_batch([tokens])
            return self.tokenizer.decode(results[0].hypotheses[0])
        except Exception as e:
            print(f"Erro ao traduzir texto: {e}")
            return None

def process_csv(input_file, output_file, model_path, tokenizer_path, batch_size=8):
    """Processa um arquivo CSV com poemas para tradução
    
    Args:
        input_file: caminho do CSV de entrada
        output_file: caminho do CSV de saída
        model_path: caminho do modelo .pt
        tokenizer_path: caminho do tokenizer .model
        batch_size: tamanho do lote para tradução
    """
    try:
        # Carrega os dados
        df = pd.read_csv(input_file)
        
        if 'original_poem' not in df.columns:
            raise ValueError("CSV deve conter coluna 'original_poem'")
        
        # Inicializa tradutor
        translator = PoemTranslator(model_path, tokenizer_path)
        
        # Tradução com barra de progresso
        translations = []
        for i in tqdm(range(0, len(df), batch_size), desc="Traduzindo"):
            batch = df['original_poem'].iloc[i:i+batch_size].tolist()
            
            for poem in batch:
                try:
                    translated = translator.translate_poem(poem) if pd.notna(poem) else ""
                    translations.append(translated if translated else "")
                except Exception as e:
                    print(f"Erro no lote {i}:{i+batch_size} - {e}")
                    translations.extend([""] * len(batch))
                    break
                
            time.sleep(0.1)  # Pausa para evitar sobrecarga
        
        # Adiciona resultados e salva
        df['translated_by_TA'] = translations[:len(df)]  # Garante tamanho igual
        df.to_csv(output_file, index=False)
        print(f"Tradução concluída! Salvo em {output_file}")
        
    except Exception as e:
        print(f"Erro no processamento: {e}")
        raise

if __name__ == "__main__":
    # Configurações
    CONFIG = {
        "input_csv": "../poemas/frances_ingles_poems_teste.csv",
        "output_csv": "../poemas/OpenNMT/frances_ingles_poems_OpenNMT.csv",
        "model_path": "nllb-200-1.3B-onmt.pt",
        "tokenizer_path": "flores200_sacrebleu_tokenizer_spm.model",
        "batch_size": 4  # Reduzido para maior estabilidade
    }
    
    # Executa o processo
    process_csv(**CONFIG)