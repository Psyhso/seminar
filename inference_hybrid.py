from hybrid_sim import wsd_hybrid, lemmatize

def inference(_context_words, target_word):

    context_words = list(map(lemmatize, _context_words.split()))
    best, scores = wsd_hybrid(context_words, target_word)

    print(f'Предсказанное значение: {best}')
    print(f'Косинусная близость: ')
    if scores:
        for k, v in scores.items():
            print(f" {k}: {v:.3f}")

if __name__ == '__main__':

    while True:

        target_word = input('Значение какого слова узнать: ').lower().strip()
        if target_word == 'выход':
            break
        _context_words = input('Контекст, в котором встретилось слово: ')
        print()

        inference(_context_words, target_word)
