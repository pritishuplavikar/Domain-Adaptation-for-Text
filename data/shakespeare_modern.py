from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
from vocab import Vocab
from utils import normalize_string, get_idx_sentence
import numpy as np
from constants import *

class ShakespeareModern(Dataset):
	def __init__(self, train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path):
		self.train_shakespeare_path = train_shakespeare_path
		self.test_shakespeare_path = test_shakespeare_path

		self.train_modern_path = train_modern_path
		self.test_modern_path = test_modern_path

		vocab = Vocab('ShakespeareModern')
		with open(self.train_shakespeare_path) as f:
			data = f.readlines()
		for idx, sentence in enumerate(data):
			sentence = normalize_string(sentence)
			vocab.add_sentence(sentence)
			data[idx] = get_idx_sentence(vocab, sentence)

		data.sort(key=len, reverse=True)
		max_len = len(data[0])
		sample = np.ndarray((len(data), max_len, 1, 1))
		for idx, sentence in enumerate(data):
			sample[idx, :len(sentence), 0, 0] = sentence
			sample[idx, len(sentence):, 0, 0] = [PAD_token] * (max_len - len(sentence))

		# print (max_len, sample[18000], sample.shape)

	def __getitem__(self, index):
		pass

	def __len__(self):
		pass

train_shakespeare_path = '../dataset/train.original.nltktok'
test_shakespeare_path = '../dataset/test.original.nltktok'
train_modern_path = '../dataset/train.modern.nltktok'
test_modern_path = '../dataset/test.modern.nltktok'
sm = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path)
