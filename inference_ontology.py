import pymorphy3
from ruwordnet import RuWordNet

from ontology_sim import  lemmatize, get_synsets_for_word, bfs_distance

wn = RuWordNet()
morph = pymorphy3.MorphAnalyzer()


def predict(word, context_words, max_depth=6):
    target_synsets = wn.get_synsets(lemmatize(word))
    if not target_synsets:
        return None, {}

    context_synsets = []
    for w in context_words:
        context_synsets.extend(get_synsets_for_word(w))

    if not context_synsets:
        return None, {}

    scores = {}
    for syn in target_synsets:
        total = sum(bfs_distance(syn, ctx, max_depth) for ctx in context_synsets)
        scores[syn] = total / len(context_synsets)

    best = min(scores, key=scores.get)
    return best, scores

if __name__ == '__main__':
    while True:
        print()
        word = input("Слово: ").strip().lower()
        if word == 'выход':
            break

        # Проверяем есть ли слово в RuWordNet
        synsets = wn.get_synsets(lemmatize(word))
        if not synsets:
            print(f"Слово '{word}' не найдено в RuWordNet.")
            continue

        # Показываем доступные значения
        print(f"Найдено значений в RuWordNet: {len(synsets)}")
        for s in synsets:
            print(f"  • {s.title}")

        context_input = input("Контекст (слова через пробел): ").strip().lower()
        context = context_input.split()

        if not context:
            print("Контекст не может быть пустым.")
            continue

        best, scores = predict(word, context)

        if best is None:
            print("Не удалось определить значение — слова контекста не найдены в RuWordNet.")
            continue

        print()
        print(f"Слово:    {word}")
        print(f"Контекст: {context}")
        print()
        print("Расстояния по графу (меньше = ближе):")
        for syn, dist in sorted(scores.items(), key=lambda x: x[1]):
            bar = '█' * max(0, int((6 - dist) * 4))
            marker = ' ← победитель' if syn.id == best.id else ''
            print(f"  {syn.title:<35} {dist:.2f}  {bar}{marker}")
        print()
        print(f"➜ Предсказанное значение: {best.title}")
        if best.definition:
            print(f"  Определение: {best.definition}")