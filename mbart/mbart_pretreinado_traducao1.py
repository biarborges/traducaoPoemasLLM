#ate funciona, mas o 2 é melhor
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

input = """
terre
tu rentres. tu quittes le rivage. tu retournes en terre. les amers quittent la mer.
soudain cette moitié du monde qui était en mer redevient terre – forêts, champs, campagne. a
son tour celle-ci devient l’océan. tu reviens au monde des vivants comme un grec débarqué
tournait le dos à l’inféconde. l’immensité se fait solide, moissonneuse, verte et blonde,
guéable. les nuages sont utile. tu écartes les buissons de la lisière, rentres dans le bois,
retournes à l’épais – l’impénétrable. la forêt de chênes chante.
en même temps c’est le temps, le double régime chaque moitié est le tout, dans
l’indivsion.
celle de la sérénité hölderlinienne: l’oubli de la menace, le vaste, la pérennité, le pour-
toujours du s’entr’aimer multiple, pareil au spectacle quand le monde se donne en spectacle,
l’oisiveté léopardienne; c’est quand les champs et les eaux, les forêts et les fleurs, les nuages
et les neiges assonent dans le zèle des saisons.
avec celle-ci: repoussé, pressenti, ulcérant, le contre-courant funèbre, le complot du
destin, affliction et nuisance, la conspiration de la perte, voici la morition des proches, la
contagion des maux, l’acerbe érosion, la calomnie générale, l’abréviation de la vie,
l’encombre, la terre périmée, l’extermination du passé, le périr.
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