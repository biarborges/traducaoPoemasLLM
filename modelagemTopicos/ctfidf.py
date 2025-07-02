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
CAMINHO_CSV = "results/ingles_portugues/reference/poemas_com_topicos_reference.csv"
PASTA_SAIDA = "results"
COLUNA_POEMAS = "translated_poem"  # "original_poem", "translated_poem", "translated_by_TA"

IDIOMA_ORIGEM = "en_XX"  # "fr_XX", "pt_XX", "en_XX"
IDIOMA_DESTINO = "pt_XX"  # "fr_XX", "pt_XX", "en_XX"
IDIOMA_PROC = "pt_XX"

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
# 2. DEFINIÇÃO DAS FUNÇÕES
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


# =======================================================================
# 3. BLOCO PRINCIPAL
# =======================================================================

if __name__ == '__main__':

    # Garante que a pasta de saída exista
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # --- Carregamento e filtragem ---
    print(f"📖 Carregando dados de '{CAMINHO_CSV}'...")
    df = pd.read_csv(CAMINHO_CSV)
    df = df[(df["src_lang"] == IDIOMA_ORIGEM) & (df["tgt_lang"] == IDIOMA_DESTINO)].reset_index(drop=True)
    poemas = df[COLUNA_POEMAS].astype(str).tolist()
    print(f"Encontrados {len(poemas)} poemas para análise.")

    # --- Pré-processamento ---
    print("🧹 Iniciando pré-processamento dos textos...")
    mapa_idioma_nltk = {"pt_XX": "portuguese", "en_XX": "english", "fr_XX": "french"}
    idioma_nltk = mapa_idioma_nltk[IDIOMA_PROC]

    stopwords_personalizadas = set(stopwords.words(idioma_nltk))
    if IDIOMA_PROC == "fr_XX":
        stopwords_personalizadas.update([
            "le", "la", "les", "un", "une", "jean", "john", "kaku", "lorsqu", "jusqu", "sai", "congnois",
            "mme", "williams", "non", "tatactatoum", "aucun", "rien", "worsted", "sandwich", "prononciation",
            "sûrement", "oui", "nao", "not", "não", "this", "that", "lover", "lorenzo", "oliver", "tão",
            "translation", "english", "weep", "poetic", "vanished"
        ])
    elif IDIOMA_PROC == "pt_XX":
        stopwords_personalizadas.update([
            "o", "a", "os", "as", "um", "uma", "eu", "tu", "ele", "ela", "nós", "vós",
            "eles", "elas", "voce", "nao", "algum", "bedlam", "quão", "quao"
        ])
    elif IDIOMA_PROC == "en_XX":
        stopwords_personalizadas.update([
            "the", "a", "an", "and", "but", "or", "so", "to", "of", "in", "for", "on",
            "at", "peter", "john", "mary", "jane", "kaku", "thee", "thy"
        ])

    nlp = carregar_modelo_spacy(IDIOMA_PROC)

    poemas_limpos = [preprocessar_texto(p, nlp, stopwords_personalizadas) for p in tqdm(poemas, desc="Processando poemas")]

    # Adiciona os poemas limpos no DataFrame
    df['preprocessed'] = poemas_limpos

    # ---------------------------
    # ATENÇÃO: a coluna 'topic' deve existir no DataFrame antes de rodar o agrupamento abaixo.
    # Se não existir, faça a clusterização antes para gerar os tópicos!
    # ---------------------------

    if 'topic' not in df.columns:
        raise ValueError("Coluna 'topic' não encontrada no DataFrame. Rode clustering para gerar os tópicos antes.")

    # Agrupa poemas por tópico concatenando os textos
    docs_por_topico = df.groupby('topic')['preprocessed'].apply(lambda textos: " ".join(textos)).reset_index()

    # Cria TF-IDF para os documentos (cada documento é um tópico)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,1))
    tfidf_matrix = vectorizer.fit_transform(docs_por_topico['preprocessed'])

    feature_names = np.array(vectorizer.get_feature_names_out())

    # Função para pegar as top 10 palavras por tópico
    def top_palavras_por_topico(tfidf_vec, feature_names, top_n=10):
        arr = tfidf_vec.toarray().flatten()
        sorted_indices = np.argsort(arr)[::-1]
        top_indices = sorted_indices[:top_n]
        return feature_names[top_indices]

    # Salva as palavras por tópico em arquivo txt
    with open(f"{PASTA_SAIDA}/topicos_cTFIDF_{TITLE}.txt", "w", encoding="utf-8") as f:
        for i, row in docs_por_topico.iterrows():
            top_words = top_palavras_por_topico(tfidf_matrix[i], feature_names)
            f.write(f"Tópico {row['topic']}: {', '.join(top_words)}\n")

    print("✅ Palavras-chave por tópico salvas.")

    # --- Visualizações ---

    topic_freq = df['topic'].value_counts().sort_index()

    # Gráfico de pizza
    plt.figure(figsize=(8,8))
    plt.pie(topic_freq, labels=[f"Topic {i}" for i in topic_freq.index], autopct='%1.1f%%', startangle=140)
    plt.title("Distribuição Percentual dos Tópicos")
    plt.savefig(f"{PASTA_SAIDA}/grafico_pizza_{TITLE}.png")
    plt.close()

    # Gráfico de barras
    plt.figure(figsize=(10,6))
    topic_freq.plot(kind='bar')
    plt.xlabel('Tópicos')
    plt.ylabel('Número de Poemas')
    plt.title('Distribuição de Poemas por Tópico')
    plt.savefig(f"{PASTA_SAIDA}/grafico_barras_topicos_{TITLE}.png")
    plt.close()

    # Nuvem de palavras geral (todos poemas limpos juntos)
    texto_total = " ".join(poemas_limpos)
    wc = WordCloud(width=1200, height=600, background_color='white', colormap='viridis').generate(texto_total)
    wc.to_file(f"{PASTA_SAIDA}/nuvem_palavras_geral_{TITLE}.png")

    print("✅ Visualizações geradas.")
