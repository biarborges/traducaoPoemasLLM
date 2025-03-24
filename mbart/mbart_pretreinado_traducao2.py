import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do mBART para francês -> inglês
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)

# Poema de teste em francês
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

# Separar por linhas para evitar truncamento
versos = poema_original.split("\n")

traducao_completa = []

# Traduzir verso por verso
for verso in versos:
    texto_com_prefixo = f">>en<< {verso.strip()}"  # Adicionar prefixo da língua
    encoded = tokenizer(texto_com_prefixo, return_tensors="pt", truncation=True, padding=True, max_length=512)
    encoded = {key: value.to(device) for key, value in encoded.items()}  # Mover para GPU

    with torch.no_grad():
        generated_tokens = model.generate(**encoded, max_length=512, num_beams=5)

    traducao = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    traducao_completa.append(traducao)

# Unir os versos traduzidos
poema_traduzido = "\n".join(traducao_completa)

# Exibir resultado
print("\n=== Poema Original ===\n")
print(poema_original)
print("\n=== Tradução ===\n")
print(poema_traduzido)
