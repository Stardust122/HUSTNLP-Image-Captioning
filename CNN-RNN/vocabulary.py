import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter


# 词汇表类
class Vocabulary(object):

    # 初始化类并设置参数
    def __init__(self,
                 vocab_threshold,
                 vocab_file='./vocab.pkl',
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 annotations_file='../cocoapi/annotations/captions_train2014.json',
                 vocab_from_file=False):

        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    # 获取词汇表，如果文件存在则加载，否则重新构建
    def get_vocab(self):
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    # 构建词汇表
    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    # 初始化词汇表
    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    # 向词汇表中添加单词
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    # 从标注文件中提取并添加词汇
    def add_captions(self):
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    # 根据单词返回对应的索引，若单词不存在则返回<unk>的索引
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    # 返回词汇表的长度
    def __len__(self):
        return len(self.word2idx)
