import torch
from torch.utils.data.dataset import Dataset
from data.vocab import Vocab
from data.utils import normalize_string, get_idx_sentence
import numpy as np
from data.constants import *

class ShakespeareModern(Dataset):
	def __init__(self, train_domain_A_path, test_domain_A_path, train_domain_B_path, test_domain_B_path, name='ShakespeareModern', mode='train'):
		self.train_domain_A_path = train_domain_A_path
		self.test_domain_A_path = test_domain_A_path

		self.train_domain_B_path = train_domain_B_path
		self.test_domain_B_path = test_domain_B_path

		self.vocab = Vocab(name)
		self.mode = mode

		self.domain_A_max_len = 0
		self.domain_B_max_len = 0

		self.train_domain_A_data = self.load_and_preprocess_data(self.train_domain_A_path, domain='A')
		self.test_domain_A_data = self.load_and_preprocess_data(self.test_domain_A_path, domain='A')

		self.train_domain_B_data = self.load_and_preprocess_data(self.train_domain_B_path, domain='B')
		self.test_domain_B_data = self.load_and_preprocess_data(self.test_domain_B_path, domain='B')

		# self.max_len = 0

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

		if (domain == 'A'):
			self.domain_A_max_len = max(self.domain_A_max_len, max_len)
		else:
			self.domain_B_max_len = max(self.domain_B_max_len, max_len)

		self.max_len = max(self.domain_A_max_len, self.domain_B_max_len)

		padded_sequences = np.ndarray((self.max_len, len(data), 1))
		for idx, sentence in enumerate(data):
			padded_sequences[:len(sentence), idx, 0] = sentence
			padded_sequences[len(sentence):, idx, 0] = [PAD_token] * (self.max_len - len(sentence))

		return torch.from_numpy(padded_sequences.astype(np.int64))

	def get_addn_feats(self, sentence):
		net_score = 0
		domain_A_count = 0
		domain_B_count = 0
		sent_len = 0
		for word in sentence:
			word = word[0].item()
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
			return self.test_domain_A_data[:, index, :], self.get_addn_feats(self.test_domain_A_data[:, index, :]), self.test_domain_B_data[:, index, :], self.get_addn_feats(self.test_domain_B_data[:, index, :])
		else:
			return self.train_domain_A_data[:, index, :], self.get_addn_feats(self.train_domain_A_data[:, index, :]), self.train_domain_B_data[:, index, :], self.get_addn_feats(self.train_domain_B_data[:, index, :])

	def __len__(self):
		if self.mode == 'test':
			return max(self.test_domain_A_data.size(1), self.test_domain_B_data.size(1))
		else:
			return max(self.train_domain_A_data.size(1), self.train_domain_B_data.size(1))