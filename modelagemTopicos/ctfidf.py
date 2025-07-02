import pandas as pd
import os
import spacy
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =======================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =======================================================================

TITLE = "original"
CAMINHO_CSV = "results/ingles_portugues/original/poemas_com_topicos_original.csv"
PASTA_SAIDA = "results"
COLUNA_POEMAS = "original_poem"
IDIOMA_ORIGEM = "en_XX"
IDIOMA_DESTINO = "pt_XX"
IDIOMA_PROC = "en_XX"

correcoes_lemas = {
    "conheçar": "conhecer",
    "escrevar": "escrever",
    "falecerar": "falecer",
    "pensaríar": "pensar",
    "odeiar": "odiar",
    "deuse": "deuses",
    "vivir": "viver",
    "écrir": "écrire",
}

normalizacao_lemas = {
    "neiger": "neige",
    "ensoleillé": "soleil",
    "pluvieux": "pluie",
    "aimé": "amour",
    "aimer": "amour",
    "amoureux": "amour",
    "cœur": "coeur",
    "cri": "crier",
    "glorieux": "gloire",
    "tromperie": "tromper",
    "chrétiens": "chrétien",
    "sauraient": "savoir",
    "ressurgiremos": "ressurgir",
    "falhou": "falhar",
    "podias": "poder",
    "amo": "amar",
    "amado": "amar",
    "perdoo": "perdoar",
    "inteligente": "inteligência",
    "odio": "odiar",
    "morto": "morte",
    "daddy": "dad",
    "hidden": "hide",
    "vaguely": "vague",
    "writing": "write",
}

# =======================================================================
# 2. FUNÇÕES
# =======================================================================

def carregar_modelo_spacy(idioma: str):
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
        tokens_processados = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2:
                lema_corrigido = correcoes_lemas.get(token.lemma_.lower(), token.lemma_.lower())
                lema_final = normalizacao_lemas.get(lema_corrigido, lema_corrigido)
                tokens_processados.append(lema_final)
        tokens = tokens_processados
    else:
        tokens = word_tokenize(texto.lower())

    tokens_filtrados = [
        token for token in tokens
        if token.isalpha() and token not in stopwords_custom and len(token) > 2
    ]
    return " ".join(tokens_filtrados)

def top_palavras_com_pesos(tfidf_vec, feature_names, top_n=10):
    arr = tfidf_vec.toarray().flatten()
    sorted_indices = arr.argsort()[::-1][:top_n]
    return [(feature_names[i], arr[i]) for i in sorted_indices]


# =======================================================================
# 3. EXECUÇÃO PRINCIPAL
# =======================================================================

if __name__ == '__main__':

    os.makedirs(PASTA_SAIDA, exist_ok=True)

    print(f"📖 Carregando dados de '{CAMINHO_CSV}'...")
    df = pd.read_csv(CAMINHO_CSV)
    df = df[(df["src_lang"] == IDIOMA_ORIGEM) & (df["tgt_lang"] == IDIOMA_DESTINO)].reset_index(drop=True)
    poemas = df[COLUNA_POEMAS].astype(str).tolist()
    print(f"Encontrados {len(poemas)} poemas para análise.")

    print("🧹 Iniciando pré-processamento dos textos...")

    mapa_idioma_nltk = {"pt_XX": "portuguese", "en_XX": "english", "fr_XX": "french"}
    idioma_nltk = mapa_idioma_nltk[IDIOMA_PROC]

    stopwords_personalizadas = set(stopwords.words(idioma_nltk))
    if IDIOMA_PROC == "fr_XX":
        stopwords_personalizadas.update([
            "le", "la", "les", "un", "une", "jean", "john", "kaku", "lorsqu", "jusqu", "sai",
            "congnois", "mme", "williams", "non", "tatactatoum", "aucun", "rien", "worsted",
            "sandwich", "prononciation", "sûrement", "oui", "nao", "not", "não", "this", "that",
            "lover", "lorenzo", "oliver", "tão", "translation", "english", "weep", "poetic", "vanished"
        ])
    elif IDIOMA_PROC == "pt_XX":
        stopwords_personalizadas.update(["o", "a", "os", "as", "um", "uma", "eu", "tu", "ele", "ela",
            "nós", "vós", "eles", "elas", "voce", "nao", "algum", "bedlam", "quão", "quao"])
    elif IDIOMA_PROC == "en_XX":
        stopwords_personalizadas.update(["the", "a", "an", "and", "but", "or", "so", "to", "of",
            "in", "for", "on", "at", "peter", "john", "mary", "jane", "kaku", "thee", "thy","thou"])

    nlp = carregar_modelo_spacy(IDIOMA_PROC)

    poemas_limpos = [preprocessar_texto(p, nlp, stopwords_personalizadas) for p in tqdm(poemas, desc="Processando poemas")]

    # adiciona coluna com poemas preprocessados
    df['preprocessed'] = poemas_limpos

    # --------------------------------------
    # Verifica se a coluna 'topic' existe no df
    if 'topic' not in df.columns:
        raise ValueError("A coluna 'topic' não foi encontrada no DataFrame. Ela é necessária para esta análise.")

    # Agrupa poemas por tópico concatenando os textos
    docs_por_topico = df.groupby('topic')['preprocessed'].apply(lambda textos: " ".join(textos)).reset_index()

    # Conta a quantidade de poemas por tópico (incluindo outliers -1)
    contagem_por_topico = df['topic'].value_counts().to_dict()

    # Cria TF-IDF para os documentos (cada documento é um tópico)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
    tfidf_matrix = vectorizer.fit_transform(docs_por_topico['preprocessed'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Salva as palavras por tópico com pesos e quantidade de poemas
    with open(f"{PASTA_SAIDA}/topicos_{TITLE}.txt", "w", encoding="utf-8") as f:
        for i, row in docs_por_topico.iterrows():
            top_words_pesos = top_palavras_com_pesos(tfidf_matrix[i], feature_names)
            qtd_poemas = contagem_por_topico.get(row['topic'], 0)
            f.write(f"Topic {row['topic']} (Quantity of poems: {qtd_poemas}):\n")
            for palavra, peso in top_words_pesos:
                f.write(f"  {palavra}: {peso:.4f}\n")
            f.write("\n")

    print("✅ Palavras-chave por tópico salvas.")

    # --- Visualizações ---

    # Remove outliers (tópico -1) para gráficos
    df_sem_outliers = df[df['topic'] != -1]
    docs_por_topico_sem_outliers = docs_por_topico[docs_por_topico["topic"] != -1].reset_index(drop=True)
    tfidf_matrix_sem_outliers = tfidf_matrix[docs_por_topico["topic"] != -1]


    # Frequência de tópicos para gráfico pizza e barra (sem outliers)
    topic_freq = df_sem_outliers['topic'].value_counts().sort_index()

    # Gráfico de pizza
    plt.figure(figsize=(8,8))
    plt.pie(topic_freq, labels=[f"Topic {i}" for i in topic_freq.index], autopct='%1.1f%%', startangle=140)
    plt.title("Percentage Distribution of Topics")
    plt.savefig(f"{PASTA_SAIDA}/grafico_pizza_{TITLE}.png")
    plt.close()

    # Gráfico de barras
    def plot_top_words_todos_topicos(tfidf_matrix, feature_names, docs_por_topico, top_n=10, max_topicos=5):
        num_topicos = min(len(docs_por_topico), max_topicos)
        largura_barra = 0.8
        espacamento = top_n + 2  # espaço entre os grupos de tópicos

        plt.figure(figsize=(20, 8))

        positions = []
        labels = []

        for i in range(num_topicos):
            tfidf_vec = tfidf_matrix[i].toarray().flatten()
            sorted_indices = np.argsort(tfidf_vec)[::-1][:top_n]
            top_words = feature_names[sorted_indices]
            top_scores = tfidf_vec[sorted_indices]

            pos = np.arange(top_n) + i * espacamento
            plt.bar(pos, top_scores, width=largura_barra, label=f'Topic {docs_por_topico.loc[i, "topic"]}')
            
            positions.extend(pos)
            labels.extend(top_words)

        plt.xticks(positions, labels, rotation=90)
        plt.ylabel('Weight')
        plt.title(f'Top {top_n} words by topic')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{PASTA_SAIDA}/top_palavras_todos_topicos_{TITLE}.png")
        plt.close()

    # --- Chamada da função para gerar gráfico de top palavras por tópico ---
    plot_top_words_todos_topicos(
    tfidf_matrix=tfidf_matrix_sem_outliers,
    feature_names=feature_names,
    docs_por_topico=docs_por_topico_sem_outliers,
    top_n=10,
    max_topicos=10
    )


    # Nuvem de palavras geral (todos poemas limpos juntos)
    texto_total = " ".join(poemas_limpos)
    wc = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate(texto_total)
    wc.to_file(f"{PASTA_SAIDA}/nuvem_palavras_geral_{TITLE}.png")

    print("✅ Visualizações geradas.")
