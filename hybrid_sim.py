from ruwordnet import RuWordNet
import pymorphy3
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

wn = RuWordNet()
morph = pymorphy3.MorphAnalyzer()
model = KeyedVectors.load_word2vec_format('./dataset/model.bin', binary=True)


def get_neighbors(synset):
    return (
        list(synset.hypernyms) +
        list(synset.hyponyms)  +
        list(synset.meronyms)  +
        list(synset.holonyms)  +
        list(synset.related)
    )

def lemmatize(word):
    return morph.parse(word)[0].normal_form

def extract_lemmas(synset):
    """Извлекает все уникальные леммы из синсета."""
    return list({sense.name.lower() for sense in synset.senses})

def build_sense_inventory_from_graph(word, max_neighbors_depth=1):
    """
    Для каждого синсета слова собирает слова-ассоциации:
    леммы из самого синсета + леммы из его соседей (глубина 1).
    Возвращает словарь {title: [words]}.
    """
    inventory = {}
    synsets = wn.get_synsets(word)
    for syn in synsets:
        title = syn.title  # уникальное название синсета
        assoc_words = set()
        # Добавляем собственные леммы синсета
        assoc_words.update(extract_lemmas(syn))
        # Добавляем леммы соседей
        for neighbor in get_neighbors(syn):
            assoc_words.update(extract_lemmas(neighbor))
        # Фильтруем: оставляем только те слова, для которых есть вектор в модели
        valid_words = [w for w in assoc_words if word_to_vec(w) is not None]
        if valid_words:
            inventory[title] = valid_words
    return inventory

def word_to_vec(word):
    """Поиск вектора слова, пробуя разные части речи."""
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
        if f"{word}_{pos}" in model:
            return model[f"{word}_{pos}"]
    return None

def get_mean_vector(words):
    vecs = [word_to_vec(w) for w in words if word_to_vec(w) is not None]
    return np.mean(vecs, axis=0) if vecs else None

def wsd_hybrid(context_words, target_word):
    """
    Гибридный метод: автоматический инвентарь из графа + word2vec.
    """
    # Строим инвентарь прямо сейчас (можно кешировать)
    sense_inv = build_sense_inventory_from_graph(target_word)
    if not sense_inv:
        return None, {}

    context_vec = get_mean_vector(context_words)
    if context_vec is None:
        return None, {}

    scores = {}
    for sense_title, assoc_words in sense_inv.items():
        sense_vec = get_mean_vector(assoc_words)
        if sense_vec is not None:
            scores[sense_title] = float(cosine_similarity(
                context_vec.reshape(1, -1),
                sense_vec.reshape(1, -1)
            )[0][0])

    best = max(scores, key=scores.get) if scores else None
    return best, scores


if __name__ == '__main__':
    test_cases = [
        ('замок', ['башня', 'средневековый', 'стена', 'рыцарь'],   'СРЕДНЕВЕКОВЫЙ ЗАМОК'),
        ('замок', ['ключ', 'дверь', 'открыть', 'запирать'],        'ЗАМОК ДЛЯ ЗАПИРАНИЯ'),
        ('ключ',  ['родник', 'источник', 'вода'],                  'ВОДНЫЙ ИСТОЧНИК'),
        ('ключ',  ['замок', 'дверь', 'скважина'],                  'КЛЮЧ К ЗАМКУ'),
        ('ключ',  ['нота', 'музыка', 'мелодия'],                   'МУЗЫКАЛЬНЫЙ КЛЮЧ'),
        ('ключ',  ['шифр', 'код', 'секрет'],                       'КЛЮЧ ШИФРОВАНИЯ'),
    ]

    print("Гибридный метод (автоинвентарь из графа + w2v):")
    for target, ctx, true_title in test_cases:
        best, scores = wsd_hybrid(ctx, target)
        ok = '✅' if best == true_title else '❌'
        print(f"{target} | контекст: {ctx} | истина: {true_title} | предсказано: {best} {ok}")
        if scores:
            for k,v in scores.items():
                print(f"  {k}: {v:.3f}")