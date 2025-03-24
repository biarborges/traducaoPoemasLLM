import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar o tokenizador e o modelo pré-treinado para tradução do francês para o inglês
model_name = "facebook/mbart-large-50-one-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Definir o idioma de origem (francês)
tokenizer.src_lang = "fr_XX"

# Texto em francês para tradução
texto_frances = "Bonjour, comment ça va ?"

# Tokenizar o texto de entrada
inputs = tokenizer(texto_frances, return_tensors="pt").to(device)

# Obter o ID do idioma de destino (inglês)
tgt_lang_id = tokenizer.lang2id["en_XX"]

# Gerar a tradução (usando forced_bos_token_id para garantir que a tradução será para o inglês)
translated = model.generate(**inputs, forced_bos_token_id=tgt_lang_id).to(device)

# Decodificar a tradução de volta para texto
translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

# Exibir a tradução
print(f"Texto original (francês): {texto_frances}")
print(f"Tradução (inglês): {translated_text}")