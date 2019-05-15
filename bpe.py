import re, collections

def get_stats(vocab):
    """Compute frequencies of adjacent pairs of symbols."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    replacement = ''.join(pair)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(replacement, word)
        v_out[w_out] = v_in[word]
    return v_out


def encode(orig, bpe_codes):
    """Encode word based on list of BPE merge operations, which are applied consecutively"""

    word = tuple(orig) + ('</w>',)
    print("__word split into characters:__ <tt>{}</tt>".format(word))

    pairs = set([b for b in zip(word[:-1], word[1:])])

    if not pairs:
        return orig

    iteration = 0
    while True:
        iteration += 1
        print("__Iteration {}:__".format(iteration))

        print("bigrams in the word: {}".format(pairs))
        bigram = min(pairs, key=lambda pair: bpe_codes.get(pair, float('inf')))
        print("candidate for merging: {}".format(bigram))
        if bigram not in bpe_codes:
            print("__Candidate not in BPE merges, algorithm stops.__")
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        print("word after merging: {}".format(word))
        if len(word) == 1:
            break
        else:
            pairs = set([b for b in zip(word[:-1], word[1:])])

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>', ''),)

    return word

if __name__ == "__main__":
    orig_data = {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}
    train_data = {}
    for orig in orig_data:
        word = ' '.join(tuple(orig) + ('</w>',))
        train_data[word] = orig_data[orig]

    bpe_codes = {}
    bpe_codes_reverse = {}

    num_merges = 10

    for i in range(num_merges):
        print("### Iteration {}".format(i + 1))
        pairs = get_stats(train_data)
        best = max(pairs, key=pairs.get)
        train_data = merge_vocab(best, train_data)

        bpe_codes[best] = i
        bpe_codes_reverse[best[0] + best[1]] = best

        print("new merge: {}".format(best))
        print("train data: {}".format(train_data))

    encode("lowest", bpe_codes)
