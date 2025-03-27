import csv

def count_words_in_csv(csv_file):
    # Inicializando variáveis para contar as palavras
    original_poem_word_count = 0
    translated_poem_word_count = 0
    
    # Abrindo o arquivo CSV
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # Usando DictReader para acessar as colunas pelo nome
        for row in reader:
            # Contando palavras na coluna original_poem
            original_poem = row['original_poem']
            original_poem_word_count += len(original_poem.split())
            
            # Contando palavras na coluna translated_poem
            translated_poem = row['translated_poem']
            translated_poem_word_count += len(translated_poem.split())
    
    # Exibindo o número total de palavras em ambas as colunas
    print(f"Total de palavras no 'original_poem': {original_poem_word_count}")
    print(f"Total de palavras no 'translated_poem': {translated_poem_word_count}")

# Exemplo de uso
csv_file = '../traducaoPoemasLLM/poemas/poemas300/frances_ingles_poems.csv'  # Substitua pelo seu caminho do CSV
count_words_in_csv(csv_file)
