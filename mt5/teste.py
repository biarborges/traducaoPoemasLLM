from transformers import MT5ForConditionalGeneration, T5TokenizerFast
import torch

# Carregar o modelo MT5 e o tokenizer
model_name = "google/mt5-small"  # Use "google/mt5-base" ou "google/mt5-large" para modelos maiores
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5TokenizerFast.from_pretrained(model_name)

# Configurar para GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir a língua de origem e destino
SRC_LANG = "fr"  # Código para o francês
TGT_LANG = "en"  # Código para o inglês

# Função para traduzir um texto
def traduzir_texto(texto_origem):
    # Adicionar o prefixo de tradução ao texto
    prompt = f"translate {SRC_LANG} to {TGT_LANG}: {texto_origem}"
    
    # Tokenizar o texto
    tokens = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    # Gerar a tradução
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=tokens,
            max_length=100,  # Ajuste este valor conforme necessário
            num_beams=5,  # Usar busca em feixe para melhorar a qualidade
            early_stopping=True,  # Parar a geração quando o modelo estiver confiante
            no_repeat_ngram_size=2,  # Evitar repetições de n-gramas
            temperature=0.7,  # Controla a criatividade da geração
            top_k=50,  # Limita o vocabulário às top-k palavras
            top_p=0.95  # Usa amostragem nucleada (nucleus sampling)
        )

    # Decodificar a saída
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Teste com uma frase simples
texto_teste = "Bonjour tout le monde."
traducao_teste = traduzir_texto(texto_teste)
print("Texto Original:", texto_teste)
print("Tradução:", traducao_teste)