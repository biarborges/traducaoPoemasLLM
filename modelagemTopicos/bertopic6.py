import pandas as pd
import torch
import spacy
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
from umap import UMAP
import os
import time

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
CAMINHO_CSV = "frances_ingles_poems.csv"
COLUNA_POEMAS = "original_poem"
LINGUA_SPACY = "fr_core_news_sm"  # pt_core_news_sm, fr_core_news_sm, en_core_web_sm
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"
DIRETORIO_SAIDA = "resultados_frances_ingles"

# Parâmetros do BERTopic
NR_TOPICOS = 10  # Pode ser um número ou 'auto'

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def carregar_spacy(model_name: str):
    """Carrega o modelo spaCy de forma segura."""
    try:
        print(f"Carregando modelo spaCy: {model_name}...")
        return spacy.load(model_name)
    except OSError:
        print(f"❌ Erro: Modelo spaCy '{model_name}' não encontrado.")
        print(f"Por favor, execute: python -m spacy download {model_name}")
        exit()

def preprocessar_textos(textos: list, nlp) -> list:
    """Limpa e pré-processa uma lista de textos usando spaCy."""
    poemas_limpos = []
    for doc in tqdm(nlp.pipe(textos), total=len(textos), desc="Pré-processando textos"):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2
        ]
        poemas_limpos.append(" ".join(tokens))
    return poemas_limpos

def salvar_topicos_txt(topic_model: BERTopic, path: str):
    """Salva os tópicos e suas palavras em um arquivo de texto."""
    print(f"Salvando tópicos legíveis em: {path}")
    with open(path, "w", encoding="utf-8") as f:
        for topic_num in topic_model.get_topic_freq().Topic:
            if topic_num == -1:
                continue
            palavras = topic_model.get_topic(topic_num)
            palavras_str = ", ".join([p[0] for p in palavras])
            f.write(f"Tópico {topic_num}: {palavras_str}\n")

def gerar_visualizacoes(topic_model: BERTopic, embeddings: list, topics: list, df: pd.DataFrame):
    """Gera e salva todas as visualizações do modelo."""
    print("Gerando e salvando visualizações...")
    
    # Gráfico de barras das palavras dos tópicos
    fig_bar = topic_model.visualize_barchart(top_n_topics=NR_TOPICOS, n_words=10, height=400)
    pio.write_image(fig_bar, os.path.join(DIRETORIO_SAIDA, "grafico_barras_topicos.png"), scale=2)

    # Gráfico de distribuição UMAP 2D
    print("Gerando projeção UMAP 2D para visualização...")
    # Criamos um UMAP *novo* para 2D, pois o do BERTopic é 5D por padrão.
    umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.0, random_state=42)
    projecao_2d = umap_2d.fit_transform(embeddings)

    df_umap = pd.DataFrame(projecao_2d, columns=["x", "y"])
    df_umap["Tópico"] = [f"Tópico {t}" if t != -1 else "Outlier" for t in topics]
    
    fig_umap = px.scatter(
        df_umap[df_umap["Tópico"] != "Outlier"],
        x="x", y="y", color="Tópico",
        title="Distribuição de Tópicos (UMAP 2D)",
        labels={"x": "UMAP-1", "y": "UMAP-2"}
    )
    fig_umap.update_traces(marker=dict(size=5))
    pio.write_image(fig_umap, os.path.join(DIRETORIO_SAIDA, "grafico_dispersao_umap.png"), scale=2)

    # Gráfico de distribuição de tópicos (frequência)
    frequencia = df["topic"].value_counts(normalize=True).rename_axis('Tópico').reset_index(name='Proporção')
    frequencia = frequencia[frequencia["Tópico"] != -1].sort_values(by="Tópico")
    frequencia["Tópico"] = frequencia["Tópico"].astype(str)

    fig_dist = px.bar(
        frequencia, x="Tópico", y="Proporção",
        title="Distribuição de Documentos por Tópico",
        text_auto=".2%"
    )
    fig_dist.update_layout(yaxis_title="Proporção de Documentos", xaxis_title="Tópico")
    pio.write_image(fig_dist, os.path.join(DIRETORIO_SAIDA, "grafico_distribuicao_topicos.png"), scale=2)

# ==============================================================================
# FLUXO PRINCIPAL
# ==============================================================================

def main():
    """Executa o pipeline completo de modelagem de tópicos."""
    
    # Cria diretório de saída
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
    
    # Carrega modelo spaCy
    nlp = carregar_spacy(LINGUA_SPACY)

    # Carrega e pré-processa os dados
    print(f"Carregando dataset de: {CAMINHO_CSV}")
    df = pd.read_csv(CAMINHO_CSV)
    poemas_originais = df[COLUNA_POEMAS].dropna().astype(str).tolist()
    poemas_limpos = preprocessar_textos(poemas_originais, nlp)
    
    # Configura dispositivo para PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo para embeddings: {device.upper()}")
    
    # Gera embeddings
    print(f"Carregando modelo de embedding: {MODELO_EMBEDDING}")
    embedding_model = SentenceTransformer(MODELO_EMBEDDING, device=device)
    embeddings = embedding_model.encode(poemas_limpos, show_progress_bar=True)
    
    # Treina modelo BERTopic
    print("Treinando modelo BERTopic...")
    topic_model = BERTopic(language="multilingual", nr_topics=NR_TOPICOS)
    topics, probs = topic_model.fit_transform(poemas_limpos, embeddings)
    
    # Salva resultados
    df["topic"] = topics
    df["preprocessed_poem"] = poemas_limpos
    
    csv_path = os.path.join(DIRETORIO_SAIDA, "poemas_com_topicos.csv")
    print(f"Salvando CSV completo em: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    salvar_topicos_txt(topic_model, os.path.join(DIRETORIO_SAIDA, "lista_topicos.txt"))
    
    # Gera as visualizações
    gerar_visualizacoes(topic_model, embeddings, topics, df)
    
    print("\n✅ Processo finalizado com sucesso!")
    print(f"Todos os resultados foram salvos na pasta: '{DIRETORIO_SAIDA}'")

if __name__ == "__main__":
    main()