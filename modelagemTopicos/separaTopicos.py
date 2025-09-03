import re
import os

entrada = "modelagemTopicos/results/ingles_portugues/topicos.txt"
saida_dir = "modelagemTopicos/results/ingles_portugues/"

# Criar pasta de saída caso não exista
os.makedirs(saida_dir, exist_ok=True)

# Carregar arquivo
with open(entrada, "r", encoding="utf-8") as f:
    linhas = f.readlines()

print(f"✅ Arquivo carregado com {len(linhas)} linhas")

# Estrutura: {topico_num: {modelo: [linhas]}}
dados = {}
modelo_atual = None
topico_atual = None

for linha in linhas:
    linha_strip = linha.strip()
    if not linha_strip:
        continue

    # Detecta modelo
    if re.match(r'^[A-Za-z].*:$', linha_strip) and not linha_strip.startswith("Topic"):
        modelo_atual = linha_strip[:-1]
        continue

    # Detecta tópico
    match_topic = re.match(r'^Topic (\d+)', linha_strip)
    if match_topic:
        topico_atual = int(match_topic.group(1))
        if topico_atual not in dados:
            dados[topico_atual] = {}
        dados[topico_atual][modelo_atual] = []
        continue

    # Adiciona linha ao modelo dentro do tópico
    if modelo_atual and topico_atual is not None:
        dados[topico_atual][modelo_atual].append(linha)

# Salvar arquivos por tópico
for t, modelos in dados.items():
    path_saida = os.path.join(saida_dir, f"topico{t}.txt")
    with open(path_saida, "w", encoding="utf-8") as f:
        for modelo, linhas_modelo in modelos.items():
            f.write(f"{modelo}:\n")
            f.writelines(linhas_modelo)
            f.write("\n------------------------------------------------\n\n")

print(f"✅ Arquivos salvos em {saida_dir}")
