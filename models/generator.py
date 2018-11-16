import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from models.encoder import Encoder
from models.decoder import Decoder
from torch.autograd import Variable

class Generator(nn.Module):
	def __init__(self, embedding_layer, model_config, train_config, vocab_size, max_len, mode='train'):
		super(Generator, self).__init__()
		self.mode = mode
		self.train_config = train_config
		self.model_config = model_config
		self.vocab_size = vocab_size
		self.max_len = max_len
		self.embedding_layer = embedding_layer
		self.encoder = Encoder(self.model_config['embedding_size'], self.model_config['hidden_dim'], batch_size=self.train_config['batch_size'])
		self.decoder = Decoder(self.model_config['embedding_size'], self.model_config['hidden_dim'], vocab_size, max_len, batch_size=self.train_config['batch_size'])
		# if self.mode == 'train':
		# 	self.encoder.train()
		# 	self.decoder.train()
		# else:
		# 	self.encoder.eval()
		# 	self.decoder.eval()

	def forward(self, input):

		self.encoder.hidden = self.encoder.init_hidden()
		encoder_outputs, encoder_hidden = self.encoder(self.embedding_layer(input).view(input.size(0), self.train_config['batch_size'], -1))

		decoder_input = self.get_new_dec_input(SOS_token).cuda()
		decoder_hidden = encoder_hidden

		decoded_embeddings = []

		for di in range(input.size(0)):

			decoder_input = self.embedding_layer(decoder_input).view(1, self.train_config['batch_size'], -1)
			decoder_output, decoder_hidden = self.decoder(
				decoder_input, decoder_hidden, encoder_outputs)

			decoded_embeddings.append(decoder_output)

			topv, topi = decoder_output.topk(1)
			decoded_token = topi.squeeze().detach()

			decoder_input = self.get_new_dec_input(decoded_token).cuda()

			if decoded_token == EOS_token:
				break

		return torch.cat(decoded_embeddings, 0)

	def get_new_dec_input(self, token_idx):
		
		decoder_input = torch.zeros(self.vocab_size).unsqueeze(0)
		decoder_input[0, token_idx] = 1.0

		return decoder_input