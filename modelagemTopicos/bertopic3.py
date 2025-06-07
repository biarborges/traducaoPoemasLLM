import pandas as pd
from bertopic import BERTopic
import torch
from sentence_transformers import SentenceTransformer

# 1. Carrega os poemas
df = pd.read_csv("../traducaoPoemasLLM/modelagemTopicos/frances_ingles_poems.csv")  # exemplo
poemas = df["original_poem"].astype(str).tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Carrega o modelo de embeddings multilíngue
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# 3. Gera os embeddings
embeddings = embedding_model.encode(poemas, show_progress_bar=True)

# 4. Cria o BERTopic com embeddings (sem CountVectorizer)
topic_model = BERTopic(language="multilingual")

# 5. Treina o modelo
topics, probs = topic_model.fit_transform(poemas, embeddings)

# 6. Adiciona os tópicos ao DataFrame
df["topic"] = topics

# 7. Salva os resultados
df.to_csv("frances_ingles_poems_topicos.csv", index=False)
#topic_model.save("modelos/bertopic_embeddings_fr_pt")

# (Opcional) Visualização
topic_model.visualize_topics().show()
