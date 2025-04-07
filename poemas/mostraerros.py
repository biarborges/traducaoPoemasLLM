import csv

# Nome do arquivo CSV
csv_file = "../traducaoPoemasLLM/poemas/googleTradutor/frances_ingles_test_com_TA.csv"

# Nome do arquivo de saída
output_file = "erros_poemas.txt"

# Inicializar o contador de linhas
line_number = 0

# Abrir o arquivo de saída em modo de escrita
with open(output_file, "w", encoding="utf-8") as out_file:
    # Escrever cabeçalho no arquivo de saída
    out_file.write("Linhas com erro no arquivo CSV:\n\n")
    
    # Ler o arquivo CSV linha por linha e verificar o número de colunas
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        
        # Pular o cabeçalho
        next(reader, None)
        
        for row in reader:
            line_number += 1
            # Verificar o número de colunas
            if len(row) != 5:
                # Escrever no arquivo de saída
                out_file.write(f"Erro na linha {line_number}:\n")
                out_file.write(f"Conteúdo da linha: {row}\n")
                out_file.write(f"Número de colunas: {len(row)}\n")
                
                # Mostrar o verso (ajuste se necessário)
                original_poem = row[0]  # Verso original, altere se a coluna for diferente
                out_file.write(f"Verso do poema com erro: {original_poem}\n\n")
    
    print(f"Erros salvos em '{output_file}'")
