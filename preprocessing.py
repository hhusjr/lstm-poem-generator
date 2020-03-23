"""
训练诗歌数据预处理文件
@author Junru Shen
"""
from tqdm import tqdm

from utils import log


class Preprocessing:
    def __init__(self, train_poems_location):
        self._fh = open(train_poems_location, 'r', encoding='UTF-8')
        self.cleaned_data = ''
        self.words = []
        self._word2id = None
        self._id2word = None

    def preprocess(self):
        log('Running preprocessing...')
        self._clean()
        self._load_words()
        self._encode()

    def word2id(self, word):
        return self._word2id.get(word, len(self.words) - 1)

    def id2word(self, id):
        return self._id2word.get(id)

    def get_poems(self):
        return self.cleaned_data.split(']')

    def _clean(self):
        log('Reading and cleaning training file...')
        with self._fh as f:
            for line in tqdm(f):
                cur = line.strip() + ']'
                cur = cur.split(':')[1]
                # 使其作出的诗风格为五言诗
                if len(cur) <= 5:
                    continue
                if cur[5] == '，':
                    self.cleaned_data += cur

    def _load_words(self):
        _word_freq_mapping = {}
        # 词频统计
        log('Getting word frequency map...')
        for word in tqdm(sorted(list(self.cleaned_data))):
            if word in _word_freq_mapping:
                _word_freq_mapping[word] += 1
            else:
                _word_freq_mapping[word] = 1

        # 去除低频词
        log('Removing low-frequency words...')
        remove = []
        for word in tqdm(_word_freq_mapping):
            if _word_freq_mapping[word] <= 2:
                remove.append(word)
        for word in remove:
            del _word_freq_mapping[word]

        # 获得词语
        self.words = list(_word_freq_mapping.keys())
        self.words.append(' ')

    def _encode(self):
        log('Encoding words to ID...')
        self._word2id = {c: i for i, c in enumerate(self.words)}
        self._id2word = {i: c for i, c in enumerate(self.words)}
