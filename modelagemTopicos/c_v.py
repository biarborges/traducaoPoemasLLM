import pandas as pd
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configurações ---
CAMINHO_CSV = "results/ingles_portugues/chatGPTPrompt1/poemas_com_topicos_chatGPTPrompt1.csv"
COLUNA_POEMAS = "translated_by_TA"  # coluna com texto original (não processado)
IDIOMA_PROC = "pt_XX"

# Correções e normalizações de lemas (se quiser pode incluir mais)
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

# --- Funções ---

def carregar_modelo_spacy(idioma: str):
    modelos = {
        "pt_XX": "pt_core_news_md",
        "fr_XX": "fr_core_news_md",
        "en_XX": "en_core_web_md"
    }
    if idioma not in modelos:
        raise ValueError(f"Idioma '{idioma}' não suportado pelo spaCy.")
    print(f"✔️ Carregando modelo spaCy para {idioma} ({modelos[idioma]})...")
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
    return tokens_filtrados  # Retorna lista de tokens, não string

# --- Execução principal ---

if __name__ == "__main__":
    # Carrega CSV
    df = pd.read_csv(CAMINHO_CSV)
    print(f"Total de poemas: {len(df)}")

    # Carregar spaCy
    nlp = carregar_modelo_spacy(IDIOMA_PROC)

    # Carregar stopwords nltk para o idioma
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    idioma_nltk = {"pt_XX": "portuguese", "en_XX": "english", "fr_XX": "french"}[IDIOMA_PROC]
    stopwords_personalizadas = set(stopwords.words(idioma_nltk))

    # Pré-processar poemas (lista de listas de tokens)
    poemas_tokenizados = []
    for poema in tqdm(df[COLUNA_POEMAS].astype(str), desc="Pré-processando poemas"):
        tokens = preprocessar_texto(poema, nlp, stopwords_personalizadas, usar_lematizacao=True)
        poemas_tokenizados.append(tokens)

    # Criar dicionário gensim
    dictionary = Dictionary(poemas_tokenizados)
    print(f"Dicionário com {len(dictionary)} tokens únicos.")

    # Defina seus grupos e tópicos (exemplo para 3 grupos só, adapte conforme seu caso)
    topic_groups = {
                "ChatGPT Prompt 1": [
            ["casa", "branco", "pequeno", "azul", "rio", "água", "olho", "luz", "ilha", "poder"],
            ["amor", "beleza", "verdade", "deus", "morte", "mundo", "dor", "olho", "vida", "verdadeiro"],
            ["pensamento", "mão", "dia", "noite", "cabeça", "sentido", "olhar", "olho", "sol", "alegria"],
    }

    # Calcular coerência para cada grupo
    for grupo, topicos in topic_groups.items():
        cm = CoherenceModel(
            topics=topicos,
            texts=poemas_tokenizados,
            dictionary=dictionary,
            coherence='c_v'
        )
        media = cm.get_coherence()
        print(f"\nGrupo {grupo} — Coerência média c_v: {media:.4f}")
        scores_topicos = cm.get_coherence_per_topic()
        for i, score in enumerate(scores_topicos):
            print(f"  Tópico {i}: {score:.4f}")
