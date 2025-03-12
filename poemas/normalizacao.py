#ingles_portugues
#ingles_frances
import pandas as pd

# Função para normalizar o poema
def normalize_poem(text):
    # Verificar se o valor é uma string
    if not isinstance(text, str):
        return ""  # Retornar uma string vazia se não for uma string
    
    # Converter para minúsculas
    text = text.lower()
    
    # Dividir o texto em linhas
    lines = text.splitlines()
    
    # Remover linhas em branco e espaços extras dentro das linhas
    cleaned_lines = []
    for line in lines:
        line = line.strip()  # Remove espaços no início e no final da linha
        if line:  # Verifica se a linha não está vazia
            cleaned_lines.append(' '.join(line.split()))  # Remove espaços extras dentro da linha
    
    # Juntar as linhas com uma única quebra de linha
    return '\n'.join(cleaned_lines)

# Carregar o arquivo CSV
input_file = 'corvo.csv'
df = pd.read_csv(input_file)

# Aplicar a normalização às colunas 'original_poem' e 'translated_poem'
df['original_poem'] = df['original_poem'].apply(normalize_poem)
df['translated_poem'] = df['translated_poem'].apply(normalize_poem)

# Salvar o arquivo CSV normalizado
output_file = 'corvo_normalizado.csv'
df.to_csv(output_file, index=False, encoding='utf-8')

print(f"Arquivo normalizado salvo em: {output_file}")