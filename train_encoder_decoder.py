import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import models.encoder
import models.decoder
from data.shakespeare_modern import ShakespeareModern
from data.utils import idx_to_sent
from tqdm import tqdm
import numpy as np

train_shakespeare_path = './dataset/train.original.nltktok'
test_shakespeare_path = './dataset/test.original.nltktok'
train_modern_path = './dataset/train.modern.nltktok'
test_modern_path = './dataset/test.modern.nltktok'


def train(model_config, num_epochs=10, batch_size=1, base_lr=0.001):
	mode = 'train'
	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)




model_config = {
	'embedding_size': 300,
	'hidden_dim': 256
}
train(model_config)