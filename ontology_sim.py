# =============================================================================
# WSD через онтологический граф RuWordNet
# =============================================================================

# Алгоритм: BFS по рёбрам графа онтологии.
# Для каждого синсета целевого слова считаем среднее расстояние
# до синсетов слов контекста. Побеждает ближайший синсет.
# =============================================================================

from collections import deque
import pymorphy3
from ruwordnet import RuWordNet
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

wn = RuWordNet()
morph = pymorphy3.MorphAnalyzer()


# =============================================================================
# 1. ТЕСТОВЫЕ ПРИМЕРЫ
# =============================================================================
# Чтобы узнать title синсета для нового слова:
#   for s in wn.get_synsets('слово'): print(s.id, s.title)

test_cases = [
    # Хорошие примеры — граф находит чёткие пути, расстояния различаются
    ('замок', ['башня', 'стена', 'рыцарь'],   'СРЕДНЕВЕКОВЫЙ ЗАМОК'),  # 5.12 vs 5.50
    ('замок', ['бойница', 'донжон', 'ров'],   'СРЕДНЕВЕКОВЫЙ ЗАМОК'),  # 3.50 vs 5.00 — самый чёткий
    ('ключ',  ['родник', 'источник', 'вода'], 'ВОДНЫЙ ИСТОЧНИК'),      # 2.43 vs 5.29+ — очень чёткий
    ('ключ',  ['замок', 'дверь', 'скважина'], 'КЛЮЧ К ЗАМКУ'),         # замок тянет к замку
    ('ключ',  ['нота', 'музыка', 'мелодия'],  'МУЗЫКАЛЬНЫЙ КЛЮЧ'),     # музыкальный домен
    ('ключ',  ['шифр', 'код', 'секрет'],      'КЛЮЧ ШИФРОВАНИЯ'),      # шифрование
]


# =============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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

def bfs_path(start, target, max_depth=6):
    """Возвращает кратчайший путь как список синсетов"""
    if start.id == target.id:
        return [start]
    visited = {start.id}
    queue = deque([(start, [start])])
    while queue:
        current, path = queue.popleft()
        if len(path) > max_depth:
            continue
        for neighbor in get_neighbors(current):
            if neighbor.id == target.id:
                return path + [neighbor]
            if neighbor.id not in visited:
                visited.add(neighbor.id)
                queue.append((neighbor, path + [neighbor]))
    return []

def get_synsets_for_word(word):
    return wn.get_synsets(lemmatize(word))


# =============================================================================
# 3. АЛГОРИТМ WSD
# =============================================================================

def wsd_graph(context_words, target_word, max_depth=6):
    context_synsets = []
    for word in context_words:
        context_synsets.extend(get_synsets_for_word(word))

    if not context_synsets:
        return None, {}

    target_synsets = wn.get_synsets(target_word)
    scores = {}
    for target_syn in target_synsets:
        total = sum(bfs_distance(target_syn, ctx, max_depth)
                    for ctx in context_synsets)
        scores[target_syn] = total / len(context_synsets)

    best = min(scores, key=scores.get)
    return best, scores


# =============================================================================
# 4. ВИЗУАЛИЗАЦИЯ — ТОЛЬКО ПУТИ BFS
# =============================================================================

def short_title(synset, max_len=20):
    t = synset.title or synset.id
    return t[:max_len] + ('...' if len(t) > max_len else '')

def build_path_graph(root_synset, context_synsets, max_depth=6):
    """
    Строим граф только из узлов которые лежат на путях BFS
    от root_synset до синсетов контекста.
    Никаких лишних узлов — только то что нужно для понимания.
    """
    G = nx.DiGraph()
    path_edges = set()
    ctx_nodes = set()
    all_path_nodes = {root_synset.id}

    for ctx_syn in context_synsets:
        path = bfs_path(root_synset, ctx_syn, max_depth)
        if not path:
            continue
        ctx_nodes.add(path[-1].id)
        for node in path:
            all_path_nodes.add(node.id)
            G.add_node(node.id, title=short_title(node))
        for i in range(len(path) - 1):
            G.add_edge(path[i].id, path[i+1].id)
            path_edges.add((path[i].id, path[i+1].id))

    # Убеждаемся что корень есть в графе
    if root_synset.id not in G:
        G.add_node(root_synset.id, title=short_title(root_synset))

    return G, path_edges, ctx_nodes

def draw_path_graph(ax, root_synset, context_synsets, title, avg_dist, max_depth=6):
    G, path_edges, ctx_nodes = build_path_graph(
        root_synset, context_synsets, max_depth
    )

    if len(G.nodes) <= 1:
        ax.text(0.5, 0.5, 'Пути не найдены\n(нет связей в онтологии)',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return

    # Иерархический layout — корень сверху
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.5)

    # Цвета узлов
    node_colors = []
    node_sizes  = []
    for node_id in G.nodes:
        if node_id == root_synset.id:
            node_colors.append('#E74C3C')   # красный — целевой синсет
            node_sizes.append(1200)
        elif node_id in ctx_nodes:
            node_colors.append('#2ECC71')   # зелёный — контекст
            node_sizes.append(900)
        else:
            node_colors.append('#F39C12')   # оранжевый — путь BFS
            node_sizes.append(600)

    # Цвета рёбер
    edge_colors = ['#E67E22' if (u, v) in path_edges else '#BDC3C7'
                   for u, v in G.edges]
    edge_widths = [2.5 if (u, v) in path_edges else 1.0
                   for u, v in G.edges]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors,
                           width=edge_widths,
                           arrows=True,
                           arrowsize=15,
                           alpha=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=0.95)
    labels = {n: G.nodes[n]['title'] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    legend = [
        mpatches.Patch(color='#E74C3C', label='Целевой синсет'),
        mpatches.Patch(color='#2ECC71', label='Синсет слова из контекста'),
        mpatches.Patch(color='#F39C12', label='Промежуточный узел пути'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=8)
    ax.set_title(f'{title}\nСреднее расстояние до контекста: {avg_dist:.2f}',
                 fontsize=10, fontweight='bold')
    ax.axis('off')

def visualize_comparison(target_word, context_words, title1, title2, max_depth=6):
    """
    Рисуем два подграфа рядом.
    title1, title2 — titles конкретных синсетов которые хотим сравнить.
    Чтобы узнать titles: for s in wn.get_synsets('слово'): print(s.title)
    """
    context_synsets = []
    for word in context_words:
        context_synsets.extend(get_synsets_for_word(word))

    all_synsets = {s.title: s for s in wn.get_synsets(target_word)}
    syn1 = all_synsets.get(title1)
    syn2 = all_synsets.get(title2)

    if not syn1 or not syn2:
        print(f"Не найден синсет: '{title1}' или '{title2}'")
        return

    dist1 = sum(bfs_distance(syn1, c, max_depth) for c in context_synsets) / max(len(context_synsets), 1)
    dist2 = sum(bfs_distance(syn2, c, max_depth) for c in context_synsets) / max(len(context_synsets), 1)
    winner = syn1 if dist1 < dist2 else syn2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(
        f'Обход графа RuWordNet: слово "{target_word}"\n'
        f'Контекст: {context_words}\n'
        f'Победитель (мин. расстояние): {winner.title}',
        fontsize=12, fontweight='bold'
    )

    draw_path_graph(ax1, syn1, context_synsets,
                    f'Значение 1: {syn1.title}', dist1, max_depth)
    draw_path_graph(ax2, syn2, context_synsets,
                    f'Значение 2: {syn2.title}', dist2, max_depth)

    plt.tight_layout()
    fname = f'graph_{target_word}_{"_".join(context_words[:2])}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Сохранён: {fname}")
    plt.show()


# =============================================================================
# 5. ЗАПУСК
# =============================================================================

W = 76
print("=" * W)
print(f"{'СЛОВО':<8} {'КОНТЕКСТ':<36} {'ВЕРНО':<22} {'OK':^4}")
print("=" * W)

correct = 0
for word, context, true_title in test_cases:
    best_synset, scores = wsd_graph(context, word)
    pred_title = best_synset.title if best_synset else '?'
    ok = '✅' if pred_title == true_title else '❌'
    correct += (ok == '✅')

    scores_str = ' | '.join(
        f"{s.title}:{d:.2f}"
        for s, d in sorted(scores.items(), key=lambda x: x[1])
    )
    print(f"{word:<8} {str(context)[:34]:<36} {true_title:<22} {ok:^4}")
    print(f"         Расстояния: {scores_str}")
    print()

print("=" * W)
print(f"ACCURACY: {correct}/{len(test_cases)}")
print("=" * W)
print()

print("Строю графы...")
visualize_comparison('замок', ['башня', 'стена', 'рыцарь'],
                     'СРЕДНЕВЕКОВЫЙ ЗАМОК', 'ЗАМОК ДЛЯ ЗАПИРАНИЯ')
visualize_comparison('замок', ['бойница', 'донжон', 'ров'],
                     'СРЕДНЕВЕКОВЫЙ ЗАМОК', 'ЗАМОК ДЛЯ ЗАПИРАНИЯ')
visualize_comparison('ключ',  ['родник', 'источник', 'вода'],
                     'ВОДНЫЙ ИСТОЧНИК', 'КЛЮЧ К ЗАМКУ')