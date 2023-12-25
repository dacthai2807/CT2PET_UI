def generate_sentences(words, min_num_words=2):
    n = len(words)
    result = []
    for i in range(n):
        j = i + min_num_words
        while j <= n:
            result.append(' '.join(words[i:j]))
            j += 1
    return result

print(generate_sentences('thráumềm', 8))