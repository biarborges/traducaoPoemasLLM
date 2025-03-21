import torch
from transformers import MarianMTModel, MarianTokenizer

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carregar modelo e tokenizer do MarianMT
model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"  
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

# Poema de teste em francês
poema_original = """je m’éveillai, c’était la maison natale,
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
ainsi parle aujourd’hui la vie murée dans la vie."""

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
