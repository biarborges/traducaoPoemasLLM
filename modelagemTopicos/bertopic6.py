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

# --- Configurações ---
CAMINHO_CSV = "frances_ingles_poems.csv"  # ajuste seu arquivo aqui
COLUNA_POEMAS = "original_poem"
LINGUA_SPACY = "fr_core_news_sm"  # ex: "pt_core_news_sm", "fr_core_news_sm", "en_core_web_sm"
DIRETORIO_SAIDA = "frances_ingles"

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
topic_model = BERTopic(language="multilingual")
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
#topic_model.visualize_topics().write_html(os.path.join(DIRETORIO_SAIDA, "visual_topics.html"))
#topic_model.visualize_barchart(top_n_topics=10, n_words=10, width=500, height=500).write_html(os.path.join(DIRETORIO_SAIDA, "visual_barchart.html"))

n_topicos = len(set(topics)) - (1 if -1 in topics else 0)

# Gera o gráfico com todos os tópicos
fig = topic_model.visualize_barchart(top_n_topics=n_topicos, n_words=5, width=300, height=500)

# Aumenta o tamanho da fonte dos labels
fig.update_layout(
    font=dict(
        size=14  # ajuste o tamanho conforme quiser
    ),
    yaxis=dict(
        tickfont=dict(size=16)  # aumenta o tamanho da fonte das palavras no eixo Y
    ),
    xaxis=dict(
        tickfont=dict(size=14)
    )
)


pio.write_image(fig, os.path.join(DIRETORIO_SAIDA, "barchart.png"))

#fig = topic_model.visualize_topics()
#pio.write_image(fig, os.path.join(DIRETORIO_SAIDA, "topics.png"))

#-------------------------------------------------------------------------------------------------------------------------

# Frequência dos tópicos
topic_freq = topic_model.get_topic_freq()

# --- Dados ---
labels = [f"Topic {t}" for t in topic_freq["topic"]]
sizes = topic_freq["percentual"]

# --- Plotagem ---
plt.figure(figsize=(8, 8))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',  # mostra a porcentagem com uma casa decimal
    startangle=140,
    colors=plt.cm.tab20.colors  # opcional: cores variadas
)
plt.title("Percentage distribution of topics")
plt.tight_layout()
plt.savefig(os.path.join(DIRETORIO_SAIDA, "distribuicao_topicos_pizza.png"))
plt.close()


print("✅ Processo finalizado! Veja a pasta:", DIRETORIO_SAIDA)
