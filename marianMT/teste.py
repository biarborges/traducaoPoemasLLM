from transformers import MarianMTModel, MarianTokenizer

# Carregar o último checkpoint (ou qualquer outro)
model = MarianMTModel.from_pretrained("/home/ubuntu/finetuning/marianMT_frances_ingles/checkpoint-81")
tokenizer = MarianTokenizer.from_pretrained("/home/ubuntu/finetuning/marianMT_frances_ingles/checkpoint-81")

# Testar com uma frase de exemplo
input_text = "Bonjour tout le monde"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
translated = model.generate(input_ids)

# Decodificar a tradução
output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(output_text)
