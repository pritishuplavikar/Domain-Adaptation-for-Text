import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from models.encoder import Encoder
from models.decoder import Decoder
from torch.autograd import Variable

class Generator(nn.Module):
	def __init__(self, model_config, train_config, vocab_size, max_len, mode='train'):
		super(Generator, self).__init__()
		self.mode = mode
		self.train_config = train_config
		self.model_config = model_config
		self.vocab_size = vocab_size
		self.max_len = max_len
		self.embedding_layer = nn.Embedding(self.vocab_size, model_config['embedding_size'], padding_idx=PAD_token)
		self.encoder = Encoder(self.model_config['embedding_size'], self.model_config['hidden_dim'], batch_size=self.train_config['batch_size'])
		self.decoder = Decoder(self.model_config['embedding_size'], self.model_config['hidden_dim'], vocab_size, max_len, batch_size=self.train_config['batch_size'])

	def forward(self, input):
		self.encoder.hidden = self.encoder.init_hidden()
		embeddings = self.embedding_layer(input).view(input.size(0), self.train_config['batch_size'], -1).cuda()
		encoder_outputs, encoder_hidden = self.encoder(embeddings)

		decoder_input = self.get_new_dec_input(SOS_token).cuda()
		decoder_hidden = encoder_hidden

		decoded_embeddings = []

		for di in range(self.max_len):

			decoder_input = self.embedding_layer(decoder_input).view(1, self.train_config['batch_size'], -1)
			decoder_output, decoder_hidden, decoder_attention = self.decoder(
				decoder_input, decoder_hidden, encoder_outputs)

			decoded_embeddings.append(decoder_output)

			topv, topi = decoder_output.topk(1)
			decoded_token = topi.squeeze().detach()

			decoder_input = self.get_new_dec_input(decoded_token).cuda()

		decoded_embeddings = torch.cat(decoded_embeddings, 0)

		return decoded_embeddings, decoded_embeddings.max(dim=1)[1]

	def get_new_dec_input(self, token_idx):
		decoder_input = torch.tensor([[token_idx]])

		return decoder_input
