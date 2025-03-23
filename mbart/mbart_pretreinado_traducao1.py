import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do mBART
model_name = "facebook/mbart-large-50"  # Modelo pré-treinado do mBART
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Definir idioma de origem e destino
src_lang = "fr_XX"  # Francês como idioma de origem
trg_lang = "en_XX"  # Inglês como idioma de destino

# Configurar idioma de origem
tokenizer.src_lang = src_lang

# Função para traduzir
def traduzir_texto(texto, tokenizer, model, device, trg_lang):
    # Codificar o texto de entrada
    encoded = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU

    # Gerar tradução
    with torch.no_grad():
        generated_tokens = model.generate(**encoded, max_length=512, num_beams=5, forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang])

    # Decodificar e retornar a tradução
    traducao = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return traducao

# Teste com uma frase em francês
frase_frances = "Bonjour, comment ça va ?"
traducao_ingles = traduzir_texto(frase_frances, tokenizer, model, device, trg_lang)

print(f"Frase original (fr): {frase_frances}")
print(f"Tradução (en): {traducao_ingles}")
