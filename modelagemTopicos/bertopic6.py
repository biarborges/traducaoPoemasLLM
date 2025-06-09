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

# --- Configurações ---
CAMINHO_CSV = "portugues_frances_poems.csv"  
COLUNA_POEMAS = "original_poem"
LINGUA_SPACY = "pt_core_news_sm"  # "pt_core_news_sm", "fr_core_news_sm", "en_core_web_sm"
DIRETORIO_SAIDA = "portugues_frances"

# --- Função de pré-processamento com spaCy ---
print(f"Carregando spaCy modelo: {LINGUA_SPACY} ...")
nlp = spacy.load(LINGUA_SPACY)

def preprocess(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2:
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

# --- Função para salvar Topics legíveis em txt ---
def salvar_topicos_txt(topic_model, path_txt):
    with open(path_txt, "w", encoding="utf-8") as f:
        for topic_num in topic_model.get_topic_freq().Topic:
            if topic_num == -1:  # Topic outlier
                continue
            palavras = topic_model.get_topic(topic_num)
            palavras_str = ", ".join([p[0] for p in palavras])
            f.write(f"Topic {topic_num}: {palavras_str}\n")

# --- MAIN ---

print("Carregando dataset...")
df = pd.read_csv(CAMINHO_CSV)
poemas = df[COLUNA_POEMAS].astype(str).tolist()

print("Pré-processando poemas (com spaCy)...")
poemas_limpos = []
for p in tqdm(poemas, desc="Preprocessando"):
    poemas_limpos.append(preprocess(p))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo para embeddings: {device}")

print("Carregando modelo de embeddings...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

print("Gerando embeddings...")
embeddings = embedding_model.encode(poemas_limpos, show_progress_bar=True)

print("Treinando modelo BERTopic...")
# Ajuste: min_cluster_size pequeno → mais tópicos
hdbscan_model = HDBSCAN(min_cluster_size=6, min_samples=2, metric='euclidean', prediction_data=True)

# Usa o modelo HDBSCAN no BERTopic
topic_model = BERTopic(language="multilingual", hdbscan_model=hdbscan_model, nr_topics=4)
#topic_model = BERTopic(language="multilingual")
topics, probs = topic_model.fit_transform(poemas_limpos, embeddings)


print("Adicionando Topics ao DataFrame...")
df["topic"] = topics

os.makedirs(DIRETORIO_SAIDA, exist_ok=True)

csv_path = os.path.join(DIRETORIO_SAIDA, "poemas_com_topicos.csv")
print(f"Salvando CSV em {csv_path}...")
df.to_csv(csv_path, index=False)

txt_path = os.path.join(DIRETORIO_SAIDA, "topicos.txt")
print(f"Salvando Topics em texto legível em {txt_path}...")
salvar_topicos_txt(topic_model, txt_path)

print("Gerando gráficos...")

n_topicos = len(set(topics)) - (1 if -1 in topics else 0)

# Gera o gráfico com todos os tópicos
fig = topic_model.visualize_barchart(top_n_topics=n_topicos, n_words=10, width=300, height=500)

# Aumenta o tamanho da fonte dos labels
fig.update_layout(
    font=dict(
        size=20  # ajuste o tamanho conforme quiser
    ),
    yaxis=dict(
        tickfont=dict(size=20)  # aumenta o tamanho da fonte das palavras no eixo Y
    ),
    xaxis=dict(
        tickfont=dict(size=20)
    )
)


pio.write_image(fig, os.path.join(DIRETORIO_SAIDA, "barchart.png"))

#fig = topic_model.visualize_topics()
#pio.write_image(fig, os.path.join(DIRETORIO_SAIDA, "topics.png"))

#-------------------------------------------------------------------------------------------------------------------------

# Frequência dos tópicos
topic_freq = topic_model.get_topic_freq()

# Remove outlier (-1)
topic_freq = topic_freq[topic_freq["Topic"] != -1]

# Renomeia as colunas para minúsculo
topic_freq.columns = [col.lower() for col in topic_freq.columns]  

# Adiciona coluna de percentual
total_docs = topic_freq["count"].sum()
topic_freq["percentual"] = topic_freq["count"] / total_docs * 100

# --- Dados para gráfico ---
labels = [f"Tópico {t}" for t in topic_freq["topic"]]
sizes = topic_freq["percentual"]

# --- Plotagem ---
plt.figure(figsize=(8, 8))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.tab20.colors
)
plt.title("Distribuição percentual dos tópicos")
plt.tight_layout()
plt.savefig(os.path.join(DIRETORIO_SAIDA, "distribuicao_topicos_pizza.png"))
plt.close()



print("✅ Processo finalizado! Veja a pasta:", DIRETORIO_SAIDA)
