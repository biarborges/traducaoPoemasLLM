from transformers import MarianMTModel, MarianTokenizer

# Carregar modelo treinado
model_path = "/home/ubuntu/finetuning/marianMT/marianMT_frances_ingles/checkpoint-81"
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Texto de entrada em francês
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

# Tokenizar entrada
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Gerar tradução com idioma de destino forçado para inglês
translated = model.generate(
    input_ids,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],  # Garante que o modelo sabe que está traduzindo para inglês
    max_length=200,
    num_beams=5  # Melhora a qualidade da tradução ao explorar mais possibilidades
)

# Decodificar resultado
output_text = tokenizer.decode(translated[0], skip_special_tokens=True)
print(output_text)
