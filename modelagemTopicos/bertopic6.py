import pandas as pd
from bertopic import BERTopic
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import os
import time

def etapa(nome):
    print(f"\n🔧 {nome}...")
    time.sleep(0.5)

nltk.download('stopwords')
stop_words_fr = stopwords.words('french')

output_dir = "frances_ingles"
os.makedirs(output_dir, exist_ok=True)

# 1. Carrega o dataset
etapa("Carregando o dataset")
df = pd.read_csv("/home/ubuntu/traducaoPoemasLLM/modelagemTopicos/frances_ingles_poems.csv")
poemas = df["original_poem"].astype(str).tolist()

# 2. Configura dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🚀 Usando dispositivo: {device}")

# 3. Configura vectorizer com stopwords do francês
vectorizer_model = CountVectorizer(stop_words=stop_words_fr)

# 4. Carrega modelo embeddings
etapa("Carregando o modelo de embeddings")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# 5. Gera embeddings
etapa("Gerando embeddings")
embeddings = embedding_model.encode(poemas, show_progress_bar=True)

# 6. Cria e treina o BERTopic
etapa("Treinando o BERTopic")
topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(poemas, embeddings)

# 7. Adiciona tópicos ao DataFrame
etapa("Salvando tópicos no CSV")
df["topic"] = topics
df.to_csv(os.path.join(output_dir, "frances_ingles_poems_topicos.csv"), index=False)

# 8. Salva visualizações
etapa("Gerando visualizações")
topic_model.visualize_topics().write_html(os.path.join(output_dir, "bertopic_visual_topics.html"))
topic_model.visualize_barchart(top_n_topics=10).write_html(os.path.join(output_dir, "bertopic_barchart.html"))
topic_model.visualize_heatmap().write_html(os.path.join(output_dir, "bertopic_heatmap.html"))
topic_model.visualize_hierarchy().write_html(os.path.join(output_dir, "bertopic_hierarchy.html"))

# 9. Salva tópicos em TXT
etapa("Salvando tópicos em arquivo TXT")
with open(os.path.join(output_dir, "topicos_bertopic.txt"), "w", encoding="utf-8") as f:
    f.write("Tópicos extraídos com stopwords em francês\n\n")
    topics_info = topic_model.get_topic_info()
    for topic_num in topics_info.Topic:
        if topic_num == -1:
            continue
        f.write(f"Tópico {topic_num}:\n")
        words_probs = topic_model.get_topic(topic_num)
        for word, prob in words_probs:
            f.write(f"  {word} ({prob:.4f})\n")
        f.write("\n")

# 10. Salva outliers
etapa("Salvando documentos outliers")
outliers_df = df[df["topic"] == -1]
outliers_df.to_csv(os.path.join(output_dir, "outliers_poemas.csv"), index=False)

with open(os.path.join(output_dir, "outliers_poemas.txt"), "w", encoding="utf-8") as f:
    f.write(f"Documentos classificados como OUTLIERS (tópico = -1): {len(outliers_df)}\n\n")
    for i, row in outliers_df.iterrows():
        f.write(f"Index {i}: {row['original_poem']}\n\n")

print(f"\n✅ Finalizado! Arquivos salvos no diretório {output_dir}.")
print(f"Outliers encontrados: {len(outliers_df)} (salvos em outliers_poemas.csv e outliers_poemas.txt)")
