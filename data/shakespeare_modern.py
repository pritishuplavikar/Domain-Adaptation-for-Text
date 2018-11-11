import torch
from torch.utils.data.dataset import Dataset
from data.vocab import Vocab
from data.utils import normalize_string, get_idx_sentence
import numpy as np
from data.constants import *

class ShakespeareModern(Dataset):
	def __init__(self, train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, name='ShakespeareModern', mode='train'):
		self.train_shakespeare_path = train_shakespeare_path
		self.test_shakespeare_path = test_shakespeare_path

		self.train_modern_path = train_modern_path
		self.test_modern_path = test_modern_path

		self.vocab = Vocab(name)
		self.mode = mode

		self.train_shakespeare_data = self.load_and_preprocess_data(self.train_shakespeare_path, domain='A')
		self.test_shakespeare_data = self.load_and_preprocess_data(self.test_shakespeare_path, domain='A')

		self.train_modern_data = self.load_and_preprocess_data(self.train_modern_path, domain='B')
		self.test_modern_data = self.load_and_preprocess_data(self.test_modern_path, domain='B')

	def load_and_preprocess_data(self, path, domain):
		with open(path) as f:
			data = f.readlines()

		for idx, sentence in enumerate(data):
			sentence = normalize_string(sentence)
			self.vocab.add_sentence(sentence, domain)
			data[idx] = get_idx_sentence(self.vocab, sentence)

		max_len = 0
		for sentence in data:
			max_len = max(max_len, len(sentence))

		padded_sequences = np.ndarray((len(data), max_len, 1, 1))
		for idx, sentence in enumerate(data):
			padded_sequences[idx, :len(sentence), 0, 0] = sentence
			padded_sequences[idx, len(sentence):, 0, 0] = [PAD_token] * (max_len - len(sentence))

		return torch.from_numpy(padded_sequences.astype(np.int64))

	def get_addn_feats(self, sentence):
		net_score = 0
		domain_A_count = 0
		domain_B_count = 0
		sent_len = 0
		for word in sentence:
			word = word[0][0].item()
			if not word in self.vocab.tokens:
				sent_len += 1
				word = self.vocab.idx2wrd[word]
				if word in self.vocab.domain_A_vocab and word in self.vocab.domain_B_vocab:
					net_score += self.vocab.domain_A_vocab[word] - self.vocab.domain_B_vocab[word]
				elif word in self.vocab.domain_A_vocab:
					net_score += self.vocab.domain_A_vocab[word]
					domain_A_count += 1
				elif word in self.vocab.domain_B_vocab:
					net_score -= self.vocab.domain_B_vocab[word]
					domain_B_count += 1

		return torch.Tensor([net_score, domain_A_count, domain_B_count]) / sent_len


	def __getitem__(self, index):
		if self.mode == 'test':
			return self.test_shakespeare_data[index], self.get_addn_feats(self.test_shakespeare_data[index]), self.test_modern_data[index], self.get_addn_feats(self.test_modern_data[index])
		else:
			return self.train_shakespeare_data[index], self.get_addn_feats(self.train_shakespeare_data[index]), self.train_modern_data[index], self.get_addn_feats(self.train_modern_data[index])

	def __len__(self):
		if self.mode == 'test':
			return max(len(self.test_shakespeare_data), len(self.test_modern_data))
		else:
			return max(len(self.train_shakespeare_data), len(self.train_modern_data))

# train_shakespeare_path = '../dataset/train.original.nltktok'
# test_shakespeare_path = '../dataset/test.original.nltktok'
# train_modern_path = '../dataset/train.modern.nltktok'
# test_modern_path = '../dataset/test.modern.nltktok'
# sm = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path)