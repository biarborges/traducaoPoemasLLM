import re
import importlib.util


file_path = r"C:\Users\biarb\OneDrive\UFU\Mestrado\Dissertacao\traducaoPoemasLLM\modelagemTopicos\results\ingles_portugues\topico3.txt"
dict_path = r"C:\Users\biarb\OneDrive\UFU\Mestrado\Dissertacao\traducaoPoemasLLM\modelagemTopicos\results\dicionario2.py"


spec = importlib.util.spec_from_file_location("dicionario2", dict_path)
dicionario_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dicionario_module)
equivalents = getattr(dicionario_module, "equivalents", [])


equivalence_sets = [set(map(str.lower, group)) for group in equivalents]


def are_equivalent(word1, word2):
    word1, word2 = word1.lower(), word2.lower()
    for group in equivalence_sets:
        if word1 in group and word2 in group:
            return True
    return word1 == word2  # fallback para palavras idênticas


def parse_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"-{5,}", text.strip())
    data = {}
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        name = lines[0].replace(":", "").strip()
        words = set()
        for line in lines[1:]:
            match = re.match(r"([a-zA-ZÀ-ÿ'\-]+):", line)
            if match:
                words.add(match.group(1).lower())
        data[name] = words
    return data


data = parse_file(file_path)
original_words = data.get("Original", set())
reference_words = data.get("Reference", set())


def count_overlap(set_a, set_b):
    overlap = set()
    for w1 in set_a:
        for w2 in set_b:
            if are_equivalent(w1, w2):
                overlap.add(w1 + " ↔ " + w2)
    return overlap


print("\n=== Overlap com ORIGINAL (considerando equivalentes) ===")
for model, words in data.items():
    if model == "Original":
        continue
    overlap = count_overlap(original_words, words)
    print(f"\n{model}: {len(overlap)} correspondências")
    if overlap:
        print(sorted(list(overlap)))


print("\n=== Overlap com REFERENCE (considerando equivalentes) ===")
for model, words in data.items():
    if model in ["Original", "Reference"]:
        continue
    overlap = count_overlap(reference_words, words)
    print(f"\n{model}: {len(overlap)} correspondências")
    if overlap:
        print(sorted(list(overlap)))
