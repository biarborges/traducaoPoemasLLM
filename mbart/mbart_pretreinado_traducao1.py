import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Carregar o tokenizador e o modelo pré-treinado para tradução do francês para o inglês
model_name = "facebook/mbart-large-50-one-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Definir o idioma de origem (francês) e o idioma de destino (inglês)
tokenizer.src_lang = "fr_XX"

# Texto em francês para tradução
texto_frances = "Bonjour, comment ça va ?"

# Tokenizar o texto de entrada
inputs = tokenizer(texto_frances, return_tensors="pt")

# Gerar a tradução
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("en_XX")).to(device)

# Decodificar a tradução de volta para texto
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# Exibir a tradução
print(f"Texto original (francês): {texto_frances}")
print(f"Tradução (inglês): {translated_text}")
