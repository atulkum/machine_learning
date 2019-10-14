import numpy as np
import codecs
import re
import os
from numpy import linalg as LA

def get_glove(glove_path, embd_dim):
    print("Loading GLoVE vectors from file: {}".format(glove_path))
    home = os.path.expanduser('~')
    embd_path = os.path.join(home, 'Downloads/embd.npy')
    word2id_path = os.path.join(home, 'Downloads/word2id.npy')
    if os.path.exists('~/Downloads/embd.npy'):
        embd = np.load(embd_path)
        word2id = np.load(word2id_path)
        return word2id, embd

    vocab_size = int(4e5)
    word2id = {}
    embd = np.zeros((vocab_size, embd_dim))
    for i, line in enumerate(codecs.open(glove_path, 'r', 'utf-8')):
        line = re.split('\s+', line.strip())
        word = line[0]
        embd[i] = list(map(float, line[1:]))
        embd[i] = embd[i] / LA.norm(embd[i])
        word2id[word] = i
    np.save(embd_path, embd)
    np.save(word2id_path, word2id)
    return word2id, embd

if __name__ == '__main__':
    home = os.path.expanduser('~')
    word2id, embd = get_glove(os.path.join(home,'dl_entity/glove.6B/glove.6B.50d.txt'), 50)
    id2word = {v: k for k, v in word2id.items()}
    print('glove enbedding loaded')

    word = ''
    while word != 'q':
        print ('Input a word> ')
        word = input()
        if word not in word2id:
            continue

        wid = word2id[word]
        similarities = embd[wid].dot(embd.transpose())
        sorted_nn = np.argsort(similarities)[::-1]

        for i in sorted_nn[:5]:
            if i != wid:
                print(id2word[i])

