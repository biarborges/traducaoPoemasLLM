import csv
import os

# Nome do arquivo CSV
csv_file = os.path.abspath("../traducaoPoemasLLM/poemas/poemas300/validation/frances_ingles_validation.csv")

# Inicializar o contador de poemas
poem_count = 0

# Ler o arquivo CSV e contar os poemas
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None)  # Pular o cabeçalho de forma segura
    
    for row in reader:
        # Ignorar linhas vazias ou com espaços extras
        row = [col.strip() for col in row]  # Remover espaços antes e depois
        
        # Verificar se a linha tem exatamente 2 colunas e ambas não são vazias
        if len(row) == 4 and row[0] and row[1] and row[2] and row[3]:
            poem_count += 1
            
            # Pegando as primeiras frases de cada poema (separando por nova linha)
           # original_first_sentence = row[0].split("\n")[0]  # Pega a primeira linha do poema original
           # translated_first_sentence = row[1].split("\n")[0]  # Pega a primeira linha do poema traduzido
            
          #  print(f"Poema {poem_count}:")
           # print(f"Original: {original_first_sentence}")
           # print(f"Traduzido: {translated_first_sentence}")
          #  print("---")


print(f"Total de poemas no CSV: {poem_count}")
