from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained("/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-81")
tokenizer = MarianTokenizer.from_pretrained("/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-81")

input_text = """ça ressemble
à un cartoon
tchouri comète surveillée
pour le bien de l’humanité accueillir la vie
matière organique accueillir des
champignons non hallucinogènes
japonais
de préférence
ça fait plus sérieux
en terme de cartoons
et de micro tchouri
tout est véritablement
un jeu de billes
des billes à dimensions variables
sur un tissu à textures variables
sur une langue infra-silencieuse
protection non garantie
des bulles et des micro lacs disséminés
autour desquels les chaises se renversent
que puissent se lire des lettres cunéiformes
au gré des branchages
network de bulles aquatiques
forme solide gazeuse apparente sous
ma bassine d’eau isolée en apparence
le berceau
d’un réseau
sans fil"""
input_ids = tokenizer.encode(input_text, return_tensors="pt")
translated = model.generate(input_ids)

output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(output_text)
