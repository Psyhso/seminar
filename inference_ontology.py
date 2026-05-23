# =============================================================================
# Инференс WSD через онтологический граф RuWordNet
# =============================================================================
# Пользователь вводит слово и контекст — получает предсказанное значение.
# Никакого ручного инвентаря — работает на любом слове из RuWordNet.
#
# Зависимости: pip install ruwordnet pymorphy3
# =============================================================================

from collections import deque
import pymorphy3
from ruwordnet import RuWordNet

wn = RuWordNet()
morph = pymorphy3.MorphAnalyzer()

# =============================================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def lemmatize(word):
    return morph.parse(word)[0].normal_form

def get_neighbors(synset):
    return (
        list(synset.hypernyms) +
        list(synset.hyponyms)  +
        list(synset.meronyms)  +
        list(synset.holonyms)  +
        list(synset.related)
    )

def bfs_distance(start, target, max_depth=6):
    if start.id == target.id:
        return 0
    visited = {start.id}
    queue = deque([(start, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in get_neighbors(current):
            if neighbor.id == target.id:
                return depth + 1
            if neighbor.id not in visited:
                visited.add(neighbor.id)
                queue.append((neighbor, depth + 1))
    return max_depth

def get_synsets_for_word(word):
    return wn.get_synsets(lemmatize(word))

# =============================================================================
# 2. АЛГОРИТМ
# =============================================================================

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

# =============================================================================
# 3. ИНТЕРАКТИВНЫЙ РЕЖИМ
# =============================================================================

print("=" * 50)
print("WSD через онтологический граф — интерактивный режим")
print("Работает с любым словом из RuWordNet")
print("Введите 'выход' для завершения")
print("=" * 50)

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