import pandas as pd
from OpenNMT.tokenizers import Tokenizer
from OpenNMT.models import load_model
from OpenNMT.config import load_config
from OpenNMT import Runner
import sentencepiece as spm
import time
from tqdm import tqdm

class PoemTranslator:
    def __init__(self, model_path, tokenizer_path):
        """
        Inicializa o tradutor com o modelo e tokenizer
        
        Args:
            model_path: caminho para o modelo .pt do OpenNMT
            tokenizer_path: caminho para o modelo .model do SentencePiece
        """
        # Configuração do tokenizer
        self.tokenizer = Tokenizer("sentencepiece", sp_model_path=tokenizer_path)
        
        # Configuração do modelo OpenNMT
        config = {
            "model_dir": "",
            "data": {
                "source_vocabulary": "vocab.txt",
                "target_vocabulary": "vocab.txt"
            }
        }
        
        # Carrega o modelo
        self.model = load_model(model_path, model_config=config)
        
        # Configura o runner
        self.runner = Runner(self.model, self.tokenizer)
    
    def translate_poem(self, text, src_lang='fra_Latn', tgt_lang='eng_Latn'):
        """
        Traduz um texto usando o modelo NLLB
        
        Args:
            text: texto a ser traduzido
            src_lang: código do idioma de origem (padrão: inglês)
            tgt_lang: código do idioma de destino (padrão: português)
            
        Returns:
            Texto traduzido
        """
        # Adiciona os tokens de idioma
        tagged_text = f"{src_lang} {text} {tgt_lang}"
        
        # Faz a tradução
        output = self.runner.infer(tagged_text)
        
        return output[0]['tokens']

def process_csv(input_file, output_file, model_path, tokenizer_path, batch_size=8):
    """
    Processa um arquivo CSV, traduzindo os poemas e salvando os resultados
    
    Args:
        input_file: caminho do arquivo CSV de entrada
        output_file: caminho do arquivo CSV de saída
        model_path: caminho para o modelo .pt
        tokenizer_path: caminho para o tokenizer .model
        batch_size: tamanho do lote para tradução (padrão: 8)
    """
    # Carrega os dados
    df = pd.read_csv(input_file)
    
    # Verifica se a coluna existe
    if 'original_poem' not in df.columns:
        raise ValueError("O arquivo CSV deve conter uma coluna 'original_poem'")
    
    # Inicializa o tradutor
    translator = PoemTranslator(model_path, tokenizer_path)
    
    # Lista para armazenar as traduções
    translations = []
    
    # Traduz os poemas em lotes (com barra de progresso)
    for i in tqdm(range(0, len(df), batch_size), desc="Traduzindo poemas"):
        batch = df['original_poem'].iloc[i:i+batch_size].tolist()
        
        # Traduz cada poema do lote
        for poem in batch:
            try:
                translated = translator.translate_poem(poem)
                translations.append(translated)
            except Exception as e:
                print(f"Erro ao traduzir poema: {e}")
                translations.append("")  # adiciona vazio em caso de erro
        
        # Pequena pausa para evitar sobrecarga
        time.sleep(0.1)
    
    # Adiciona as traduções ao DataFrame
    df['translated_by_TA'] = translations
    
    # Salva o resultado
    df.to_csv(output_file, index=False)
    print(f"Tradução concluída! Resultados salvos em {output_file}")

if __name__ == "__main__":
    # Configurações (ajuste conforme necessário)
    INPUT_CSV = "../poemas/frances_ingles_poems_teste.csv"          # Arquivo CSV de entrada
    OUTPUT_CSV = "../poemas/OpenNMT/frances_ingles_poems_OpenNMT.csv"  # Arquivo CSV de saída
    MODEL_PATH = "nllb-200-1.3B-onmt.pt"  # Caminho para o modelo OpenNMT
    TOKENIZER_PATH = "flores200_sacrebleu_tokenizer_spm.model"  # Caminho para o tokenizer
    
    # Executa o processo
    process_csv(INPUT_CSV, OUTPUT_CSV, MODEL_PATH, TOKENIZER_PATH)