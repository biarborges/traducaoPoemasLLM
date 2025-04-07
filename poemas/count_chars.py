import csv

def count_characters_in_csv(csv_file):
    # Inicializando variáveis para contar os caracteres
    original_poem_char_count = 0
    translated_poem_char_count = 0
    
    # Abrindo o arquivo CSV
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # Usando DictReader para acessar as colunas pelo nome
        for row in reader:
            # Contando caracteres na coluna original_poem
            original_poem = row['original_poem']
            original_poem_char_count += len(original_poem)
            
            # Contando caracteres na coluna translated_poem
            translated_poem = row['translated_poem']
            translated_poem_char_count += len(translated_poem)
    
    # Exibindo o número total de caracteres em ambas as colunas
    print(f"Total de caracteres no 'original_poem': {original_poem_char_count}")
    print(f"Total de caracteres no 'translated_poem': {translated_poem_char_count}")

# Exemplo de uso
csv_file = '../traducaoPoemasLLM/poemas/frances_ingles_poems.csv'  # Substitua pelo seu caminho do CSV
count_characters_in_csv(csv_file)
