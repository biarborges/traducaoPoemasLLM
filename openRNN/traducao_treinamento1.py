import os
import sacremoses
from subword_nmt import apply_bpe
import argparse

# Configurações corrigidas
CONFIG = {
    "source_lang": "fr",
    "target_lang": "en",
    "data_dir": "/home/ubuntu/TraducaoPoemasLLM/openRNN/opus_data_en_fr",
    "model_dir": "/home/ubuntu/TraducaoPoemasLLM/openRNN/models_en_fr",  # Deve ser diretório, não arquivo
    "use_gpu": True
}

class Translator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = sacremoses.MosesTokenizer(lang=config['source_lang'])
        self.detokenizer = sacremoses.MosesDetokenizer(lang=config['target_lang'])
        
        # Verifica e carrega BPE corretamente
        bpe_path = os.path.join(config['data_dir'], "bpe.codes")
        self.bpe = None
        if os.path.exists(bpe_path):
            print(f"Carregando BPE codes de {bpe_path}")
            try:
                # Verifica se o arquivo BPE é válido
                with open(bpe_path, 'r') as f:
                    for i, line in enumerate(f, 1):
                        parts = line.strip().split()
                        if len(parts) != 2:
                            print(f"Aviso: Linha {i} mal formatada no arquivo BPE: '{line.strip()}'")
                
                # Se passou na verificação, carrega
                with open(bpe_path, 'r') as bpe_file:
                    self.bpe = apply_bpe.BPE(bpe_file)
            except Exception as e:
                print(f"Erro ao carregar BPE: {e}")
                self.bpe = None
        
        # Encontra o modelo mais recente
        if os.path.isfile(config['model_dir']):
            # Se model_dir é na verdade um arquivo .pt
            self.model_path = config['model_dir']
        else:
            model_files = [f for f in os.listdir(config['model_dir']) 
                         if f.startswith("model_") and f.endswith(".pt")]
            if not model_files:
                raise FileNotFoundError(f"Nenhum modelo .pt encontrado em {config['model_dir']}")
            self.model_path = os.path.join(config['model_dir'], sorted(model_files)[-1])
        
        print(f"Usando modelo: {self.model_path}")

    def preprocess(self, text):
        """Pré-processa o texto para tradução"""
        tokenized = self.tokenizer.tokenize(text.strip(), return_str=True)
        if self.bpe:
            try:
                tokenized = self.bpe.process_line(tokenized)
            except Exception as e:
                print(f"Erro ao aplicar BPE: {e}")
        return tokenized

    def postprocess(self, text):
        """Pós-processa o texto traduzido"""
        if self.bpe:
            text = text.replace("@@ ", "")
        return self.detokenizer.detokenize(text.split())

    def translate(self, text, beam_size=5):
        """Traduz um texto usando o modelo treinado"""
        # Cria arquivos temporários com nomes únicos
        import tempfile
        temp_dir = "/home/ubuntu/TraducaoPoemasLLM/openRNN/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        input_path = os.path.join(temp_dir, "temp_input.txt")
        output_path = os.path.join(temp_dir, "temp_output.txt")
        
        try:
            # Pré-processamento
            preprocessed = self.preprocess(text)
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write(preprocessed + "\n")
            
            # Comando de tradução
            translate_cmd = (
                f"onmt_translate -model {self.model_path} "
                f"-src {input_path} -output {output_path} "
                f"-beam_size {beam_size} "
                f"-gpu {'0' if self.config['use_gpu'] else ''} "
                f"-replace_unk -verbose"
            )
            
            print(f"Executando: {translate_cmd}")
            os.system(translate_cmd)
            
            # Lê e pós-processa a saída
            with open(output_path, 'r', encoding='utf-8') as f:
                translated = self.postprocess(f.read().strip())
            
            return translated
        
        finally:
            # Limpa arquivos temporários
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

def main():
    parser = argparse.ArgumentParser(description='Tradutor FR-EN usando modelo RNN')
    parser.add_argument('--text', type=str, help='Texto para traduzir')
    parser.add_argument('--file', type=str, help='Arquivo para traduzir')
    parser.add_argument('--beam', type=int, default=5, help='Tamanho do beam search')
    args = parser.parse_args()

    translator = Translator(CONFIG)
    
    if args.file:
        # Modo arquivo - versão segura para grandes arquivos
        output_file = os.path.splitext(args.file)[0] + "_translated.txt"
        with open(args.file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                translated = translator.translate(line.strip(), args.beam)
                fout.write(translated + "\n")
        print(f"Tradução completa. Resultado salvo em {output_file}")
    elif args.text:
        # Modo texto único
        translated = translator.translate(args.text, args.beam)
        print(f"Texto original: {args.text}")
        print(f"Tradução: {translated}")
    else:
        # Modo interativo
        print(f"Tradutor {CONFIG['source_lang']}->{CONFIG['target_lang']} (digite 'quit' para sair)")
        while True:
            text = input("\nTexto para traduzir: ")
            if text.lower() == 'quit':
                break
            translated = translator.translate(text, args.beam)
            print(f"Tradução: {translated}")

if __name__ == "__main__":
    main()