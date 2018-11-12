import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.encoder import Encoder
from models.decoder import Decoder
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from data.shakespeare_modern import ShakespeareModern
from data.utils import idx_to_sent
from tqdm import tqdm
import numpy as np

train_shakespeare_path = './dataset/train.original.nltktok'
test_shakespeare_path = './dataset/test.original.nltktok'
train_modern_path = './dataset/train.modern.nltktok'
test_modern_path = './dataset/test.modern.nltktok'


def train(model_config, train_config):
	mode = 'train'
	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)
	vocab = dataset.vocab
	max_length = dataset.domain_A_max_len
	encoder = Encoder(model_config['embedding_size'], model_config['hidden_dim'], dataset.vocab.num_words, batch_size=train_config['batch_size']).cuda()
	# print(dataset.domain_A_max_len)
	decoder = Decoder(model_config['embedding_size'], model_config['hidden_dim'], dataset.vocab.num_words, max_length, batch_size=train_config['batch_size']).cuda()

	criterion = nn.NLLLoss().cuda()
	encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=train_config['base_lr'])
	decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=train_config['base_lr'])

	for epoch in range(train_config['num_epochs']):
		for idx, (s, s_addn_feats, m, m_addn_feats) in tqdm(enumerate(dataloader)):
			input_tensor = s.transpose(0, 1).cuda()
			target_tensor = m.transpose(0, 1).cuda()

			encoder_optimizer.zero_grad()
			decoder_optimizer.zero_grad()

			input_length = input_tensor.size(0)
			target_length = target_tensor.size(0)

			encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

			loss = 0
			print ('ip', input_tensor.size())
			encoder_output, encoder_hidden = encoder(input_tensor)
			# encoder_outputs = encoder_output[0, 0]

			decoder_input = torch.empty((train_config['batch_size'], 1)).fill_(SOS_token).type(torch.LongTensor).cuda()
			print (decoder_input.size())

			decoder_hidden = encoder_output[-1]
			print ('dec hid', decoder_hidden.size(), type(decoder_hidden))

			while decoder_input:
				decoder.hidden = decoder_hidden
				decoder_input, decoder_hidden = decoder(
					decoder_input, encoder_output)

			loss += criterion(decoder_output, target_tensor[di])
			loss.backward()

			encoder_optimizer.step()
			decoder_optimizer.step()
			
			if idx % 100 == 0:
				print('\tepoch [{}/{}], iter: {}, s_loss: {:.4f}, m_loss: {:.4f}, preds: s: {}, {}, m: {}, {}'
					.format(epoch+1, train_config['num_epochs'], idx, s_loss.item(), m_loss.item(), s_output.item(), round(s_output.item()), m_output.item(), round(m_output.item())))

		print('\tepoch [{}/{}]'.format(epoch+1, train_config['num_epochs']))


		return loss.item() / target_length		


model_config = {
	'embedding_size': 300,
	'hidden_dim': 256
}

train_config = {
	'batch_size': 1,
	'continue_train': False,
	'model_path': './shakespeare_disc.pth',
	'base_lr': 0.001,
	'num_epochs': 10
}
train_config['num_epochs'] = 3
print(train(model_config, train_config))
