import pandas as pd
import os
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.io as pio
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from umap import UMAP

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# ==============================================================================

SEED = 42

TITLE = "reference"
# original reference chatGPTPrompt1 googleTradutor maritacaPrompt1

# Caminho para o arquivo de entrada
CAMINHO_CSV = "poemas_unificados.csv"
# chatGPTPrompt1 googleTradutor maritacaPrompt1

# Pasta para salvar os resultados
PASTA_SAIDA = "results"

# Coluna do DataFrame a ser utilizada
COLUNA_POEMAS = "translated_poem"  # "original_poem", "translated_poem", "translated_by_TA"

# Definição dos idiomas de origem e destino para filtrar o CSV
IDIOMA_ORIGEM = "en_XX"  #  "fr_XX", "pt_XX", "en_XX"
IDIOMA_DESTINO = "fr_XX" #  "fr_XX", "pt_XX", "en_XX"

# Idioma para o pré-processamento (NLTK e spaCy)
IDIOMA_PROC = "fr_XX"



correcoes_lemas = {
    "conheçar": "conhecer",
    "escrevar": "escrever",
    "falecerar": "falecer",
    "pensaríar": "pensar",
}

# Adicione este dicionário
normalizacao_lemas = {
    "neiger": "neige", 
    "ensoleillé": "soleil", 
    "pluvieux": "pluie",
    "aimé": "amour",
    "aimer": "amour",
    "cœur": "coeur",
    "cri": "crier",

    "ressurgiremos": "ressurgir",
    "falhou": "falhar",
    "podias": "poder",
    "amo": "amar",
}


# ==============================================================================
# 2. DEFINIÇÃO DAS FUNÇÕES
# ==============================================================================

def carregar_modelo_spacy(idioma: str):
    """Carrega o modelo spaCy apropriado para o idioma especificado."""
    print(f"✔️ Carregando modelo spaCy para o idioma: {idioma}")
    modelos = {
        "pt_XX": "pt_core_news_md",
        "fr_XX": "fr_core_news_md",
        "en_XX": "en_core_web_md"
    }
    if idioma not in modelos:
        raise ValueError(f"Idioma '{idioma}' não suportado pelo spaCy neste script.")
    return spacy.load(modelos[idioma])


def preprocessar_texto(texto: str, nlp_model, stopwords_custom: set, usar_lematizacao=True):
    if usar_lematizacao:
        doc = nlp_model(texto)
        
        # Vamos trocar a list comprehension por um loop para maior clareza
        tokens_processados = []
        for token in doc:
            # A condição 'if' que estava na sua list comprehension vem para dentro do loop
            if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2:
                
                # Passo 1: Pega o lema e aplica o dicionário de correções
                lema_corrigido = correcoes_lemas.get(token.lemma_.lower(), token.lemma_.lower())
                
                # Passo 2: Aplica o NOVO dicionário de normalização
                lema_final = normalizacao_lemas.get(lema_corrigido, lema_corrigido)
                
                tokens_processados.append(lema_final)
        
        # Atribui o resultado à variável 'tokens' que você já usava
        tokens = tokens_processados
        
    else:
        tokens = word_tokenize(texto.lower())

    # Esta parte do seu código continua exatamente igual
    tokens_filtrados = [
        token for token in tokens
        if token.isalpha() and token not in stopwords_custom and len(token) > 2
    ]
    return " ".join(tokens_filtrados)


def salvar_topicos_legiveis(topic_model: BERTopic, caminho_arquivo: str):
    """Salva os tópicos e suas palavras-chave em um arquivo de texto."""
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        for topic_num, palavras in topic_model.get_topics().items():
            # Pula o tópico de outliers, se desejado (opcional)
            # if topic_num == -1:
            #     continue
            
            count = topic_model.get_topic_info(topic_num)["Count"].iloc[0]
            palavras_str = ", ".join([p[0] for p in palavras])
            f.write(f"Tópico {topic_num} ({count} poemas): {palavras_str}\n")

# ==============================================================================
# 3. BLOCO DE EXECUÇÃO PRINCIPAL
# ==============================================================================

# Este bloco protege o código de ser re-executado em processos filhos,
# resolvendo o erro 'RuntimeError' no Windows/macOS.
if __name__ == '__main__':

    # Garante que a pasta de saída exista
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # --- Parte 1: Carregamento e Filtragem dos Dados ---
    print(f"📖 Carregando dados de '{CAMINHO_CSV}'...")
    df = pd.read_csv(CAMINHO_CSV)
    df = df[(df["src_lang"] == IDIOMA_ORIGEM) & (df["tgt_lang"] == IDIOMA_DESTINO)].reset_index(drop=True)
    poemas = df[COLUNA_POEMAS].astype(str).tolist()
    print(f"Encontrados {len(poemas)} poemas para análise.")

    # --- Parte 2: Pré-processamento ---
    print("🧹 Iniciando pré-processamento dos textos...")
    
    # Mapeia o código de idioma para o NLTK
    mapa_idioma_nltk = {"pt_XX": "portuguese", "en_XX": "english", "fr_XX": "french"}
    idioma_nltk = mapa_idioma_nltk[IDIOMA_PROC]

    # Carrega stopwords e adiciona termos personalizados
    stopwords_personalizadas = set(stopwords.words(idioma_nltk))
    if IDIOMA_PROC == "fr_XX":
        stopwords_personalizadas.update(["le", "la", "les", "un", "une", "jean", "john", "kaku", "lorsqu", "jusqu", "sai", "congnois", "mme", "williams", "non", "tatactatoum", "aucun", "rien", "worsted"])
    elif IDIOMA_PROC == "pt_XX":
        stopwords_personalizadas.update(["o", "a", "os", "as", "um", "uma", "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas", "voce", "nao", "algum"])
    elif IDIOMA_PROC == "en_XX":
        stopwords_personalizadas.update(["the", "a", "an", "and", "but", "or", "so", "to", "of", "in", "for", "on", "at", "peter", "john", "mary", "jane", "kaku"])
    
    # Carrega o modelo spaCy UMA ÚNICA VEZ para eficiência
    nlp = carregar_modelo_spacy(IDIOMA_PROC)

    # Aplica o pré-processamento a todos os poemas
    poemas_limpos = [preprocessar_texto(p, nlp, stopwords_personalizadas) for p in tqdm(poemas, desc="Processando poemas")]

    # --- Parte 3: Geração de Embeddings ---
    print("🔗 Carregando modelo de embeddings (SentenceTransformer)...")
    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2") # paraphrase-multilingual-MiniLM-L12-v2  distiluse-base-multilingual-cased-v2
    print("🔗 Gerando embeddings para os poemas...")
    embeddings = embedding_model.encode(poemas_limpos, show_progress_bar=True)

    # --- Parte 4: Criação e Treinamento do Modelo BERTopic ---
    print("📚 Criando e treinando o modelo BERTopic...")
    
    umap_model = UMAP(random_state=SEED)
    vectorizer_model=CountVectorizer(ngram_range=(1, 1))

    topic_model = BERTopic(language="multilingual", umap_model=umap_model, vectorizer_model=vectorizer_model, representation_model=KeyBERTInspired(), embedding_model=embedding_model)
    topics, _ = topic_model.fit_transform(poemas_limpos, embeddings)

    # --- Parte 5: Análise e Salvamento dos Resultados ---
    print("📊 Adicionando tópicos ao DataFrame e salvando...")
    df["topic"] = topics
    df.to_csv(f"{PASTA_SAIDA}/poemas_com_topicos_{TITLE}.csv", index=False)

    print("📝 Salvando descrição dos tópicos em TXT...")
    salvar_topicos_legiveis(topic_model, f"{PASTA_SAIDA}/topicos_{TITLE}.txt")

    # --- Parte 6: Geração de Visualizações ---
    print("🎨 Gerando e salvando gráficos...")
    
    # Gráfico de Barras dos Tópicos
    n_topicos_reais = len(topic_model.get_topic_freq()) - (1 if -1 in topic_model.get_topic_freq().Topic.values else 0)
    fig_bar = topic_model.visualize_barchart(top_n_topics=n_topicos_reais, n_words=10, width=400, height=400)
    pio.write_image(fig_bar, f"{PASTA_SAIDA}/grafico_barras_topicos_{TITLE}.png")

    # Nuvem de Palavras Geral
    texto_total = " ".join(poemas_limpos)
    wc = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate(texto_total)
    wc.to_file(f"{PASTA_SAIDA}/nuvem_palavras_geral_{TITLE}.png")

    # Gráfico de Pizza da Distribuição dos Tópicos
    topic_freq = topic_model.get_topic_freq()
    topic_freq = topic_freq[topic_freq.Topic != -1] # Exclui outliers
    plt.figure(figsize=(8, 8))
    plt.pie(
        topic_freq["Count"],
        labels=[f"Topic {i}" for i in topic_freq["Topic"]],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title("Percentage Distribution of Topics")
    plt.savefig(f"{PASTA_SAIDA}/grafico_pizza_{TITLE}.png")
    plt.close()
    

    # --- Parte 7: Cálculo de Coerência dos Tópicos ---
    print("🧮 Preparando dados para o cálculo de coerência...")

    # Adicionamos uma barra de progresso para a tokenização
    documentos_tokenizados = [doc.split() for doc in tqdm(poemas_limpos, desc="   Tokenizando documentos")]

    dicionario = Dictionary(documentos_tokenizados)

    # Adicionamos uma barra de progresso para a criação do corpus Bag-of-Words
    corpus = [dicionario.doc2bow(doc) for doc in tqdm(documentos_tokenizados, desc="   Criando corpus BoW   ")]

    # Esta parte é rápida, não precisa de barra de progresso
    topicos_palavras = []
    for topic_num in sorted(topic_model.get_topics().keys()):
        if topic_num == -1:
            continue
        palavras = [palavra for palavra, _ in topic_model.get_topic(topic_num)]
        topicos_palavras.append(palavras)

    # Inicia o cálculo de coerência (a parte mais demorada)
    print("⏳ Calculando a coerência do modelo... Isso pode levar alguns minutos.")

    coherence_model = CoherenceModel(
        topics=topicos_palavras,
        texts=documentos_tokenizados,
        dictionary=dicionario,
        corpus=corpus,
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    print(f"✅ Coerência do Modelo c_v: {coherence_score:.4f}")
    print(f"Quantidade de Tópicos: {len(topicos_palavras)}")
    print("\n🎉 Processo concluído com sucesso!")
