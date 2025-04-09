import pandas as pd
from opennmt import tokenizers
from opennmt import models
from opennmt import inputters
from opennmt import translators
from opennmt import constants
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
        self.sp_tokenizer = spm.SentencePieceProcessor()
        self.sp_tokenizer.Load(tokenizer_path)
        
        # Configuração do modelo OpenNMT
        self.model = models.Transformer.from_config({
            "model_dir": "",
            "data": {
                "source_vocabulary": "vocab.txt",  # necessário mas não usado diretamente
                "target_vocabulary": "vocab.txt"    # necessário mas não usado diretamente
            }
        })
        
        # Carrega o modelo
        self.model.load(model_path)
        
        # Configura o tradutor
        self.translator = translators.Translator(
            model=self.model,
            tokenizer=self.sp_tokenizer,
            device=constants.DEVICE_GPU if constants.CUDA else constants.DEVICE_CPU,
            beam_size=5,
            length_penalty=0.6
        )
    
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
        output = self.translator.translate([tagged_text])
        
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
    OUTPUT_CSV = "../poemas/openNMT/frances_ingles_poems_opennmt.csv"  # Arquivo CSV de saída
    MODEL_PATH = "nllb-200-1.3B-onmt.pt"  # Caminho para o modelo OpenNMT
    TOKENIZER_PATH = "flores200_sacrebleu_tokenizer_spm.model"  # Caminho para o tokenizer
    
    # Executa o processo
    process_csv(INPUT_CSV, OUTPUT_CSV, MODEL_PATH, TOKENIZER_PATH)