import csv

# Nome do arquivo CSV
csv_file = "../traducaoPoemasLLM/poemas/googleTradutor/frances_ingles_poems_googleTradutor.csv"

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
        if len(row) == 5 and row[0] and row[1] and row[2] and row[3] and row[4]:
            poem_count += 1

print(f"Total de poemas no CSV: {poem_count}")
