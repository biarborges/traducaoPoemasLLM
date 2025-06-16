import pandas as pd
import os
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords

# ==============================================================================
# 1. CONFIGURAÇÕES E CONSTANTES (Mesmo de antes)
# ==============================================================================
CAMINHO_CSV = "chatGPTPrompt1/poemas_unificados.csv"
PASTA_SAIDA = "chatGPTPrompt1/original"
COLUNA_POEMAS = "original_poem"
IDIOMA_ORIGEM = "fr_XX"
IDIOMA_DESTINO = "en_XX"
IDIOMA_PROC = "fr_XX"

# ==============================================================================
# 2. FUNÇÕES E PRÉ-PROCESSAMENTO (Mesmo de antes)
# ==============================================================================
def carregar_modelo_spacy(idioma: str):
    print(f"✔️ Carregando modelo spaCy para o idioma: {idioma}")
    modelos = {
        "pt_XX": "pt_core_news_sm",
        "fr_XX": "fr_core_news_sm",
        "en_XX": "en_core_web_sm"
    }
    return spacy.load(modelos[idioma])

def preprocessar_texto(texto: str, nlp_model, stopwords_custom: set) -> str:
    doc = nlp_model(texto)
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2
    ]
    tokens_filtrados = [
        token for token in tokens
        if token.isalpha() and token not in stopwords_custom and len(token) > 2
    ]
    return " ".join(tokens_filtrados)

# ==============================================================================
# 3. BLOCO DE EXECUÇÃO SEGURO PARA DIAGNÓSTICO
# ==============================================================================
if __name__ == '__main__':
    # Carrega os dados normalmente
    print(f"📖 Carregando dados de '{CAMINHO_CSV}'...")
    df = pd.read_csv(CAMINHO_CSV)
    df = df[(df["src_lang"] == IDIOMA_ORIGEM) & (df["tgt_lang"] == IDIOMA_DESTINO)].reset_index(drop=True)
    poemas = df[COLUNA_POEMAS].astype(str).tolist()

    # Pré-processa normalmente
    print("🧹 Iniciando pré-processamento dos textos...")
    mapa_idioma_nltk = {"pt_XX": "portuguese", "en_XX": "english", "fr_XX": "french"}
    stopwords_personalizadas = set(stopwords.words(mapa_idioma_nltk[IDIOMA_PROC]))
    if IDIOMA_PROC == "fr_XX":
        stopwords_personalizadas.update(["le", "la", "les", "un", "une", "jean", "john", "kaku", "lorsqu", "jusqu", "sai"])
    nlp = carregar_modelo_spacy(IDIOMA_PROC)
    poemas_limpos = [preprocessar_texto(p, nlp) for p in tqdm(poemas, desc="Processando poemas")]

    # --------------------------------------------------------------------------
    # SCRIPT DE INSPEÇÃO SEGURO (NÃO USA GENSIM)
    # --------------------------------------------------------------------------
    print("\n\n--- INICIANDO INSPEÇÃO SEGURA DOS DADOS ---")
    try:
        documentos_tokenizados = [doc.split() for doc in poemas_limpos]

        # Calculando estatísticas vitais
        num_docs = len(documentos_tokenizados)
        total_tokens = sum(len(doc) for doc in documentos_tokenizados)
        maior_documento = max(len(doc) for doc in documentos_tokenizados) if documentos_tokenizados else 0
        
        # Forma segura e eficiente de contar palavras únicas sem sobrecarregar a memória
        print("   Calculando vocabulário...")
        vocabulario = set()
        for doc in tqdm(documentos_tokenizados, desc="   Analisando vocabulário"):
            vocabulario.update(doc)
        num_palavras_unicas = len(vocabulario)

        print("\n--- RESULTADOS DO DIAGNÓSTICO ---")
        print(f"Número total de documentos: {num_docs}")
        print(f"Número total de tokens (palavras): {total_tokens}")
        print(f"Comprimento do MAIOR poema (em palavras): {maior_documento}")
        print(f"Tamanho do vocabulário (palavras ÚNICAS): {num_palavras_unicas}")
        print("------------------------------------")

    except Exception as e:
        print(f"\nOcorreu um erro durante a inspeção: {e}")

    print("\nInspeção concluída. Se chegou até aqui sem travar, por favor, envie os resultados acima.")