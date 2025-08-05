import pandas as pd
import glob
import os

# --- 1. CONFIGURAÇÃO ---
caminho_para_arquivos = "poemas/openRNN" 

padrao_arquivos = os.path.join(caminho_para_arquivos, "*_*_poems_openRNN.csv")

lista_arquivos = glob.glob(padrao_arquivos)

if not lista_arquivos:
    print(f"Nenhum arquivo .csv encontrado no caminho: {caminho_para_arquivos}")
else:
    print(f"Arquivos encontrados: {len(lista_arquivos)}")
    print("\n".join([os.path.basename(f) for f in lista_arquivos]))

# Lista para armazenar cada DataFrame lido
lista_de_dfs = []

# --- 2. LOOP PARA LER E JUNTAR ---
for arquivo in lista_arquivos:
    print(f"\nLendo arquivo: {os.path.basename(arquivo)}...")
    
    # Carrega o dataset do arquivo
    df_temp = pd.read_csv(arquivo)
    
    # Adiciona o DataFrame à lista, sem nenhuma modificação
    lista_de_dfs.append(df_temp)

# --- 3. CONCATENAÇÃO FINAL ---
# Junta todos os DataFrames da lista em um único DataFrame
if lista_de_dfs:
    df_completo = pd.concat(lista_de_dfs, ignore_index=True)

    # --- 4. VERIFICAÇÃO E SALVAMENTO ---
    print("\n--- Processo de união concluído! ---")
    print("Amostra do DataFrame final (colunas originais mantidas):")
    print(df_completo.head())
    
    print("\nInformações do DataFrame:")
    df_completo.info()

    print("\nContagem de poemas por língua de origem (coluna 'src_lang'):")
    # Agora usamos 'src_lang' para a contagem
    print(df_completo['src_lang'].value_counts())

    # Salva o DataFrame unificado em um novo arquivo CSV
    caminho_saida = os.path.join(caminho_para_arquivos, "poemas_unificados.csv")
    df_completo.to_csv(caminho_saida, index=False, encoding='utf-8-sig')
    print(f"\nDataFrame unificado salvo com sucesso em: {caminho_saida}")
else:
    print("\nNenhum DataFrame foi processado. O arquivo final não foi gerado.")