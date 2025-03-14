import torch
from transformers import MarianMTModel, MarianTokenizer

# Caminhos dos arquivos
model_path = "/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-90"

# Poema de entrada
poema_original = """
je m’éveillai, c’était la maison natale,
il faisait nuit, des arbres se pressaient
de toutes parts autour de notre porte,
j’étais seul sur le seuil dans le vent froid,
mais non, nullement seul, car deux grands êtres
se parlaient au-dessus de moi, à travers moi.
l’un, derrière, une vieille femme, courbe, mauvaise,
l’autre debout dehors comme une lampe,
belle, tenant la coupe qu’on lui offrait,
buvant avidement de toute sa soif.
ai-je voulu me moquer, certes non,
plutôt ai-je poussé un cri d’amour
mais avec la bizarrerie du désespoir,
et le poison fut partout dans mes membres,
cérès moquée brisa qui l’avait aimée.
ainsi parle aujourd’hui la vie murée dans la vie.
"""

# Verificar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer
model = MarianMTModel.from_pretrained(model_path).to(device)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Preparar o texto para tradução
inputs = tokenizer(poema_original, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# Gerar tradução
translated_ids = model.generate(**inputs)

# Decodificar a tradução
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

print(translated_text)
