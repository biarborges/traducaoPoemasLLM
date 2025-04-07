import csv

# Caminhos dos arquivos CSV
csv_base = "../traducaoPoemasLLM/poemas/googleTradutor/portugues_ingles_poems_googleTradutor.csv"  # Arquivo com 300 poemas
csv_test = "../traducaoPoemasLLM/poemas/googleTradutor/portugues_ingles_test_pretreinado_googleTradutor.csv"  # Arquivo com 30 poemas
output_file = "../traducaoPoemasLLM/poemas/googleTradutor/portugues_ingles_test_com_TA.csv"  # Novo arquivo de saída

# Dicionário para armazenar {original_poem: translated_by_TA}
traducao_dict = {}

# Ler o arquivo base (300 poemas) e armazenar as traduções
with open(csv_base, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        original_poem = row["original_poem"]
        traducao_dict[original_poem] = row["translated_by_TA"]

# Processar o arquivo de 30 poemas e adicionar a tradução
with open(csv_test, mode="r", encoding="utf-8") as f_in, open(output_file, mode="w", encoding="utf-8", newline="") as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames + ["translated_by_TA"]  # Adicionar nova coluna
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)

    writer.writeheader()
    for row in reader:
        original_poem = row["original_poem"]
        row["translated_by_TA"] = traducao_dict.get(original_poem, "")  # Adiciona a tradução se existir
        writer.writerow(row)

print(f"Arquivo salvo em: {output_file}")
