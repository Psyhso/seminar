# =============================================================================
# Инференс WSD через word2vec
# =============================================================================
# Пользователь вводит слово и контекст — получает предсказанное значение.
#
# Зависимости: pip install gensim scikit-learn numpy
# Модель: ruscorpora_upos_cbow_300_20_2019.bin.gz
# =============================================================================

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from word2vec_sim import model, SENSE_INVENTORY


def word_to_vec(word):
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        if f"{word}_{pos}" in model:
            return model[f"{word}_{pos}"]
    return None

def get_mean_vector(words):
    vecs = [word_to_vec(w) for w in words if word_to_vec(w) is not None]
    return np.mean(vecs, axis=0) if vecs else None

def predict(word, context_words):
    if word not in SENSE_INVENTORY:
        return None, {}

    context_vec = get_mean_vector(context_words)
    if context_vec is None:
        return None, {}

    scores = {}
    for sense, sense_words in SENSE_INVENTORY[word].items():
        sense_vec = get_mean_vector(sense_words)
        if sense_vec is not None:
            scores[sense] = float(cosine_similarity(
                context_vec.reshape(1, -1),
                sense_vec.reshape(1, -1)
            )[0][0])

    best = max(scores, key=scores.get) if scores else None
    return best, scores


if __name__ == '__main__':
    while True:
        print()
        word = input("Слово: ").strip().lower()
        if word == 'выход':
            break

        if word not in SENSE_INVENTORY:
            print(f"Слово '{word}' не найдено в инвентаре.")
            print(f"Доступные слова: {', '.join(SENSE_INVENTORY.keys())}")
            continue

        context_input = input("Контекст (слова через пробел): ").strip().lower()
        context = context_input.split()

        if not context:
            print("Контекст не может быть пустым.")
            continue

        best, scores = predict(word, context)

        print()
        print(f"Слово:    {word}")
        print(f"Контекст: {context}")
        print()
        print("Scores:")
        for sense, score in sorted(scores.items(), key=lambda x: -x[1]):
            bar = '█' * int(score * 20) if score > 0 else ''
            print(f"  {sense:<12} {score:+.3f}  {bar}")
        print()
        print(f"➜ Предсказанное значение: {best}")