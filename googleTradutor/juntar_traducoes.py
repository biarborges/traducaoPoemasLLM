import pdfplumber
import pandas as pd

# Caminhos dos arquivos
pdf_path = "../traducaoPoemasLLM/poemas/de10em10/frances_ingles_poems_1_traducao_ingles.pdf"
csv_path = "../traducaoPoemasLLM/poemas/de10em10/frances_ingles_poems_1.csv"
new_csv_path = "../traducaoPoemasLLM/poemas/de10em10/frances_ingles_poems_1_googleTradutor.csv"

# Função para extrair os poemas do PDF
def extract_poems_from_pdf(pdf_path):
    poems = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Remover cabeçalho 'Machine Translated by Google'
                #text = text.replace("Machine Translated by Google", "").strip()
                # Separar os poemas pelo delimitador "------"
                poems.extend([p.strip() for p in text.split("----------------------------------------") if p.strip()])
    return poems

# Extrair os poemas do PDF
translated_poems = extract_poems_from_pdf(pdf_path)

# Carregar o CSV
df = pd.read_csv(csv_path)

# Verificar se as quantidades de poemas batem
if len(df) != len(translated_poems):
    raise ValueError(f"Erro: O CSV tem {len(df)} linhas, mas o PDF tem {len(translated_poems)} poemas extraídos.")

# Adicionar a nova coluna ao DataFrame
df["translated_by_TA"] = translated_poems

# Salvar o novo CSV
df.to_csv(new_csv_path, index=False)

print(f"Novo arquivo CSV salvo como: {new_csv_path}")