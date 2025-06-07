import pandas as pd
from bertopic import BERTopic
import torch
from sentence_transformers import SentenceTransformer

# 1. Carrega os poemas
df = pd.read_csv("../traducaoPoemasLLM/modelagemTopicos/frances_ingles_poems.csv")  # ajuste o caminho conforme necessário
poemas = df["original_poem"].astype(str).tolist()

# 2. Define o uso de GPU (se disponível)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Carrega o modelo de embeddings multilíngue
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# 4. Gera os embeddings
embeddings = embedding_model.encode(poemas, show_progress_bar=True)

# 5. Cria e treina o BERTopic com embeddings
topic_model = BERTopic(language="multilingual")
topics, probs = topic_model.fit_transform(poemas, embeddings)

# 6. Adiciona os tópicos ao DataFrame
df["topic"] = topics
df.to_csv("frances_ingles_poems_topicos.csv", index=False)

# 7. Salva visualizações como arquivos HTML
topic_model.visualize_topics().write_html("bertopic_visual_topics.html")
topic_model.visualize_barchart(top_n_topics=10).write_html("bertopic_barchart.html")
topic_model.visualize_heatmap().write_html("bertopic_heatmap.html")
topic_model.visualize_hierarchy().write_html("bertopic_hierarchy.html")

print("Arquivos HTML salvos. Abra no navegador local para visualizar.")
