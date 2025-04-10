import os

def remover_linhas_em_branco_csv(caminho_entrada, caminho_saida):
    # Verifica se o arquivo existe
    if not os.path.exists(caminho_entrada):
        print(f"Arquivo não encontrado: {caminho_entrada}")
        return

    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    # Mantém apenas as linhas que não estão completamente em branco
    linhas_sem_branco = [linha for linha in linhas if linha.strip() != ""]

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        f.writelines(linhas_sem_branco)

    print(f"Arquivo salvo em: {caminho_saida}")


# Exemplo de uso
entrada = "../poemas/openNMT/frances_ingles_poems_opennmt_old.csv"
saida = "../poemas/openNMT/frances_ingles_poems_opennmt.csv"

remover_linhas_em_branco_csv(entrada, saida)
