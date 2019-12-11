# -*- encoding: utf-8 -*-

import numpy as np
import sys
from io import open
reload(sys)
sys.setdefaultencoding('utf8')

class Poetry:
    def __init__(self):
        self.filename = "poetry"
        self.poetrys = self.get_poetrys()
        self.poetry_vectors, self.word_to_id, self.id_to_word = self.gen_poetry_vector()
        self.poetry_vectors_size = len(self.poetry_vectors)
        self._index_in_epoch = 0

    def get_poetrys(self):
        poetrys = list()
        f = open(self.filename, "r", encoding="utf-8")
        for line in f.readlines()[:50]:
            # print(line)
            _, content = line.strip('\n').strip().split(':')
            content = content.replace(' ', '')
            # print(_, content) # title and content
            # print(len(content)) # symbols and chinese characters
            if(not content or '_' in content or '(' in content or '（' in content or "□" in content
                    or '《' in content or '[' in content or ':' in content or '：'in content):
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content_list = content.replace('，', '|').replace('。', '|').split('|')
            # print(content_list)
            flag = True
            for sentence in content_list:
                slen = len(sentence)
                if slen == 0:
                    continue
                if slen != 5 and slen != 7:
                    flag = False
                    break;
            if flag:
                poetrys.append('[' + content + ']')
        return poetrys

    def gen_poetry_vector(self):
        words = sorted(set(''.join(self.poetrys) + ' '))
        # print(words) #sorted unicode sets
        id_to_word = {i: w for i, w in enumerate(words)}
        word_to_id = {w: i for i, w in id_to_word.items()}
        to_id = lambda word: word_to_id.get(word)
        poetry_vectors = [list(map(to_id, poetry)) for poetry in self.poetrys]
        # print(poetry_vectors)
        return poetry_vectors, word_to_id, id_to_word

    def next_batch(self, batch_size):
        assert batch_size < self.poetry_vectors_size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.poetry_vectors_size:
            np.random.shuffle(self.poetry_vectors)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        batches = self.poetry_vectors[start:end]
        # print(map(len, batches))
        x_batch = np.full((batch_size, max(map(len, batches))), self.word_to_id[' '], np.int32)
        for row in range(batch_size):
            x_batch[row, :len(batches[row])] = batches[row]
        y_batch = np.copy(x_batch)
        y_batch[:, :-1] = x_batch[:, 1:]
        y_batch[:, -1] = x_batch[:, 0]
        return x_batch, y_batch


p = Poetry()
# p.next_batch(10)
# x_batch, y_batch = p.next_batch(10)
# print(x_batch, y_batch)
