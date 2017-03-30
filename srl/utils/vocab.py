from collections import defaultdict
from itertools import count


PAD = "PADDING"
UNK = "UNKNOWN"


class Vocab(object):
    """Map between word and ID


    """
    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.is_frozen = False

    def convert(self, arg):
        if isinstance(arg, str):
            if arg in self.w2i:
                return self.w2i[arg]
            else:
                if self.is_frozen:
                    return self.w2i[UNK]
                else:
                    self.w2i[arg] = len(self.i2w)
                    self.i2w.append(arg)
        elif isinstance(arg, int):
            return self.i2w[arg]
        else:
            print arg
            raise ValueError

    def freeze(self):
        self.is_frozen = True

    @classmethod
    def build_from_corpus(cls, corpus):
        vocab = Vocab()
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            for word in sent:
                w2i[word[0]]
                vocab.i2w.append(word[0])
        vocab.w2i = dict(w2i)
        vocab.is_frozen = True

        return vocab
