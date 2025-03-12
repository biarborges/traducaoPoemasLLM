import pandas as pd
import glob
import os

# Definir o diretório correto
caminho_diretorio = os.path.abspath("../modelos/poemas300_normalizados/validation/")

# Lista de arquivos CSV no diretório especificado
arquivos_csv = glob.glob(os.path.join(caminho_diretorio, "*.csv"))

# Verificar se arquivos foram encontrados
if not arquivos_csv:
    print(f"Nenhum arquivo CSV encontrado em: {caminho_diretorio}")
    print(f"Caminho verificado: {caminho_diretorio}")
    print("Existe o diretório?", os.path.exists(caminho_diretorio))
    exit()

print("Arquivos encontrados:", arquivos_csv)

# Lista para armazenar os DataFrames
dataframes = []

# Ler e concatenar os arquivos
for arquivo in arquivos_csv:
    df = pd.read_csv(arquivo)
    dataframes.append(df)

# Concatenar tudo em um único DataFrame
resultado = pd.concat(dataframes, ignore_index=True)

# Salvar o resultado em um novo arquivo CSV
resultado.to_csv("poems_validation.csv", index=False)

print("Arquivos concatenados com sucesso em 'poems_validation.csv'!")


