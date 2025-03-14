import torch
from transformers import MarianMTModel, MarianTokenizer

# Caminhos dos arquivos
model_path = "/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-90"

# Poema de entrada
poema_original = """
la poésie ne sert à rien
quand sonne le réveil, strident, brutal
et qu’il te faut bouger ton corps et ton courage du lit à la douche, au café insipide,
au garage, à la route, au bureau inutile
la poésie ne sert à rien . . .
quand tu déneiges ton pallier,
quand on te grille la priorité, quand on te raye ta voiture,
quand tu renverses ton café quand tu oublies, quand tu rates,
quand pressent les réunions, quand tombent les deadlines,
quand tu oublies de passer prendre le pain et les enfants
la poésie ne sert à rien?
quand partent les absents
de nos mémoires vides, de nos images effacées par la distance, le temps et les soucis
quand leurs visages deviennent flous vagues, pâles ou imprécis
la poésie, sert-elle à quelque chose?
quand crache la télé, du monde les infamies les guerres, les morts, les dictatures, les peurs, la faim, la haine, le désespoir, la rage, l’horreur, la peine et les tortures
la poésie sert à quelque chose
quand sonne le réveil, sifflant, jovial
et que tu te sais vivant, présent, debout
dans ton espace peuplé de gens, de sons, de souvenirs
la poésie sert
à savourer ton café, à te foutre
des dates butoir, médiocrités et autres impératifs
elle t’entoure dans tes rires, tes pleurs et tes orgasmes
la poésie est un orgasme!
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
