import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

input = """
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

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


tokenizer.src_lang = "fr_XX"
encoded_text = tokenizer(input, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_text,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# Exibir a tradução
print(f"Texto original (francês): {input}")
print(f"Tradução (inglês): {translated_text}")