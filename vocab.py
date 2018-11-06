from constants import *

class vocab:
    def __init__(self, name):
        self.name = name
        self.removed_rare_words = False
        self.wrd2idx = {}
        self.wrd2cnt = {}
        self.idx2wrd = {constants.EOS_token: "EOS", constants.SOS_token: "SOS", constants.PAD_token: "PAD"}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.wrd2idx:
            self.wrd2idx[word] = self.num_words
            self.wrd2cnt[word] = 1
            self.idx2wrd[self.num_words] = word
            self.num_words += 1
        else:
            self.wrd2cnt[word] += 1

    def remove_rare_words(self, min_count):
        if self.removed_rare_words:
            return
        self.removed_rare_words = True
        retained_words = []
        for k, v in self.wrd2cnt.items():
            if v >= min_count:
                retained_words.append(k)

        # Reset all the  dictionaries
        self.wrd2idx = {}
        self.wrd2cnt = {}
        self.idx2wrd = {constants.EOS_token: "EOS", constants.SOS_token: "SOS", constants.PAD_token: "PAD"}
        self.num_words = 3
        for word in retained_words:
            self.add_word(word)
