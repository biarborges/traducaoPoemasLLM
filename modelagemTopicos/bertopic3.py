import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch
import spacy
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from umap import UMAP
import os
from hdbscan import HDBSCAN

# --- 1. CONFIGURAÇÕES ---
# Defina a língua que você quer analisar. Use 'fr', 'pt' ou 'en'.
LINGUA_ALVO = "fr_XX" 

# O script vai configurar o resto automaticamente com base na LINGUA_ALVO
CAMINHO_CSV = "poemas_unificados.csv"
COLUNA_POEMAS = "original_poem" 
COLUNA_LINGUA = "src_lang"

# Mapeamento para configurar o spaCy e o BERTopic
CONFIG_LINGUAS = {
    "fr_XX": {"spacy": "fr_core_news_sm", "bertopic": "french"},
    "pt_XX": {"spacy": "pt_core_news_sm", "bertopic": "portuguese"},
    "en_XX": {"spacy": "en_core_web_sm", "bertopic": "english"}
}

# Verifica se a lingua_alvo é válida
if LINGUA_ALVO not in CONFIG_LINGUAS:
    raise ValueError(f"LINGUA_ALVO inválida. Escolha uma entre: {list(CONFIG_LINGUAS.keys())}")

LINGUA_SPACY = CONFIG_LINGUAS[LINGUA_ALVO]["spacy"]
LINGUA_BERTOPIC = CONFIG_LINGUAS[LINGUA_ALVO]["bertopic"]
DIRETORIO_SAIDA = f"resultados_{LINGUA_ALVO}" # O diretório de saída será nomeado com base na língua

# --- 2. FUNÇÕES AUXILIARES ---

# Função de pré-processamento com spaCy
print(f"Carregando modelo spaCy: {LINGUA_SPACY} ...")
# Desabilitar componentes não necessários pode acelerar o carregamento e processamento
nlp = spacy.load(LINGUA_SPACY, disable=["parser", "ner"]) 

def preprocess(text):
    """Lematiza e remove stop words, pontuação e números."""
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2:
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

# Função para salvar Topics legíveis em txt
def salvar_topicos_txt(topic_model, path_txt):
    """Salva os tópicos e suas palavras-chave em um arquivo de texto."""
    with open(path_txt, "w", encoding="utf-8") as f:
        # Itera sobre os tópicos por frequência
        for topic_num in topic_model.get_topic_freq().Topic:
            if topic_num == -1:  # Ignora o tópico de outliers
                continue
            palavras = topic_model.get_topic(topic_num)
            palavras_str = ", ".join([p[0] for p in palavras])
            f.write(f"Topic {topic_num}: {palavras_str}\n")

# --- 3. SCRIPT PRINCIPAL ---

print("Carregando dataset...")
df_completo = pd.read_csv(CAMINHO_CSV)

# --- ALTERAÇÃO PRINCIPAL: FILTRANDO PELA LÍNGUA ---
print(f"Filtrando poemas para a língua: '{LINGUA_ALVO}'")
df = df_completo[df_completo[COLUNA_LINGUA] == LINGUA_ALVO].copy()

if df.empty:
    print(f"ERRO: Nenhum poema encontrado para a língua '{LINGUA_ALVO}'. Verifique o arquivo CSV e a coluna '{COLUNA_LINGUA}'.")
else:
    print(f"Encontrados {len(df)} poemas em '{LINGUA_ALVO}'.")
    poemas = df[COLUNA_POEMAS].astype(str).tolist()

    print("Pré-processando poemas com spaCy...")
    poemas_limpos = [preprocess(p) for p in tqdm(poemas, desc="Pré-processando")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo para embeddings: {device}")

    print("Carregando modelo de embeddings 'paraphrase-multilingual-MiniLM-L12-v2'...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

    print("Gerando embeddings...")
    embeddings = embedding_model.encode(poemas_limpos, show_progress_bar=True)

    print("Treinando modelo BERTopic...")
    # Agora especificamos a língua diretamente, em vez de "multilingual"
    topic_model = BERTopic(language=LINGUA_BERTOPIC, verbose=True) 
    topics, probs = topic_model.fit_transform(poemas_limpos, embeddings)

    print("Adicionando Tópicos ao DataFrame filtrado...")
    df["topic"] = topics

    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

    csv_path = os.path.join(DIRETORIO_SAIDA, f"poemas_com_topicos_{LINGUA_ALVO}.csv")
    print(f"Salvando CSV em {csv_path}...")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(DIRETORIO_SAIDA, f"topicos_{LINGUA_ALVO}.txt")
    print(f"Salvando Tópicos em texto legível em {txt_path}...")
    salvar_topicos_txt(topic_model, txt_path)

    print("Gerando gráficos...")
    # Gráfico de barras
    n_topicos_para_mostrar = len(df['topic'].unique()) - 1 # Mostra todos os tópicos exceto o de outliers
    if n_topicos_para_mostrar > 0:
        fig_bar = topic_model.visualize_barchart(top_n_topics=n_topicos_para_mostrar, n_words=10, width=400, height=300)
        pio.write_image(fig_bar, os.path.join(DIRETORIO_SAIDA, "barchart.png"))
        print(f"Gráfico de barras salvo em: {os.path.join(DIRETORIO_SAIDA, 'barchart.png')}")

    # Gráfico de pizza
    topic_freq = topic_model.get_topic_freq()
    topic_freq = topic_freq[topic_freq["Topic"] != -1]
    
    if not topic_freq.empty:
        labels = [f"Tópico {t}" for t in topic_freq["Topic"]]
        sizes = topic_freq["Count"]
        
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.viridis(range(len(labels))))
        plt.title(f"Distribuição de Tópicos para poemas em '{LINGUA_ALVO.upper()}'")
        plt.tight_layout()
        plt.savefig(os.path.join(DIRETORIO_SAIDA, "distribuicao_topicos_pizza.png"))
        plt.close()
        print(f"Gráfico de pizza salvo em: {os.path.join(DIRETORIO_SAIDA, 'distribuicao_topicos_pizza.png')}")

    print(f"✅ Processo finalizado! Veja a pasta: '{DIRETORIO_SAIDA}'")