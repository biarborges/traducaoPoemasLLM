import pandas as pd
from bertopic import BERTopic
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# Mostra etapa com tqdm
def etapa(nome):
    print(f"\n🔧 {nome}...")
    time.sleep(0.5)

# 1. Carrega os poemas
etapa("Carregando o dataset")
df = pd.read_csv("../traducaoPoemasLLM/modelagemTopicos/frances_ingles_poems.csv")  # ajuste o caminho se necessário
poemas = df["original_poem"].astype(str).tolist()

# 2. Verifica GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Usando dispositivo: {device}")

# 3. Carrega o modelo de embeddings
etapa("Carregando o modelo de embeddings")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# 4. Gera os embeddings com barra de progresso
etapa("Gerando embeddings")
embeddings = embedding_model.encode(poemas, show_progress_bar=True)

# 5. Cria e treina o BERTopic
etapa("Treinando o BERTopic")
topic_model = BERTopic(language="multilingual")
topics, probs = topic_model.fit_transform(poemas, embeddings)

# 6. Adiciona os tópicos ao DataFrame
etapa("Salvando tópicos no CSV")
df["topic"] = topics
df.to_csv("frances_ingles_poems_topicos.csv", index=False)

# 7. Salva visualizações
etapa("Gerando visualizações")
topic_model.visualize_topics().write_html("bertopic_visual_topics.html")
topic_model.visualize_barchart(top_n_topics=10).write_html("bertopic_barchart.html")
topic_model.visualize_heatmap().write_html("bertopic_heatmap.html")
topic_model.visualize_hierarchy().write_html("bertopic_hierarchy.html")

print("\n✅ Finalizado! Os arquivos foram salvos no diretório.")
