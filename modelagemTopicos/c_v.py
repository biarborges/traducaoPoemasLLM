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
CAMINHO_CSV = "modelagemTopicos/results/frances_ingles/original/poemas_com_topicos_original.csv"
CAMINHO_TOPICOS = "modelagemTopicos/results/frances_ingles/original/topicos_original.txt"
CAMINHO_SAIDA = "modelagemTopicos/results/frances_ingles/original/coerencia_topicos.csv"
COLUNA_POEMAS = "original_poem"  # Pode ser translated_by_TA, original_poem ou translated_poem
IDIOMA_PROC = "fr_XX"

# Correções e normalizações
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
    return tokens_filtrados

def carregar_topicos_arquivo(caminho_txt):
    topicos = []
    with open(caminho_txt, "r", encoding="utf-8") as f:
        for linha in f:
            if ":" in linha:
                palavras = linha.split(":")[1].strip().split(", ")
                topicos.append(palavras)
    return topicos

# --- Execução principal ---
if __name__ == "__main__":
    # Carrega CSV
    df = pd.read_csv(CAMINHO_CSV)
    print(f"Total de poemas: {len(df)}")

    # Carrega spaCy
    nlp = carregar_modelo_spacy(IDIOMA_PROC)

    # Stopwords
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    idioma_nltk = {"pt_XX": "portuguese", "en_XX": "english", "fr_XX": "french"}[IDIOMA_PROC]
    stopwords_personalizadas = set(stopwords.words(idioma_nltk))

    # Pré-processamento
    poemas_tokenizados = []
    for poema in tqdm(df[COLUNA_POEMAS].astype(str), desc="Pré-processando poemas"):
        tokens = preprocessar_texto(poema, nlp, stopwords_personalizadas, usar_lematizacao=True)
        poemas_tokenizados.append(tokens)

    # Criar dicionário gensim
    dictionary = Dictionary(poemas_tokenizados)
    print(f"Dicionário com {len(dictionary)} tokens únicos.")

    # Carregar tópicos do arquivo TXT
    topicos = carregar_topicos_arquivo(CAMINHO_TOPICOS)
    print(f"Total de tópicos carregados: {len(topicos)}")

    # Calcular coerência
    cm = CoherenceModel(
        topics=topicos,
        texts=poemas_tokenizados,
        dictionary=dictionary,
        coherence='c_v'
    )
    media = cm.get_coherence()
    scores_topicos = cm.get_coherence_per_topic()

    # Exibir resultados no console
    print(f"\n✔️ Coerência média c_v: {media:.4f}")
    for i, score in enumerate(scores_topicos):
        print(f"  Tópico {i}: {score:.4f}")

    # Salvar resultados em CSV
    resultados = pd.DataFrame({
        "Tópico": [f"Tópico {i}" for i in range(len(scores_topicos))],
        "Coerência": scores_topicos
    })
    resultados.loc[len(resultados)] = ["Média", media]
    
    print(f"🔍 Tentando salvar em: {os.path.abspath(CAMINHO_SAIDA)}")
    print(f"Arquivo será salvo com {len(resultados)} linhas")

    resultados.to_csv(CAMINHO_SAIDA, index=False, encoding="utf-8")

    print("✔ Arquivo salvo com sucesso?" , os.path.exists(CAMINHO_SAIDA))

    print(f"\n📂 Resultados salvos em: {CAMINHO_SAIDA}")
