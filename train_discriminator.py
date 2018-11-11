import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.models import Discriminator
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

	shakespeare_disc = Discriminator(model_config['embedding_size'], model_config['hidden_dim'], len(dataset.vocab)).cuda()
	shakespeare_disc.train()
	criterion = nn.BCELoss().cuda()
	optimizer = torch.optim.Adam(shakespeare_disc.parameters(), lr=base_lr,
								 weight_decay=1e-5)

	real_label = torch.ones((1,1)).cuda()
	fake_label = torch.zeros((1,1)).cuda()
	for epoch in range(num_epochs):
		for idx, (s, s_addn_feats, m, m_addn_feats) in tqdm(enumerate(dataloader)):
			s = s.squeeze(0)
			m = m.squeeze(0)

			s = Variable(s).cuda()
			s_output = shakespeare_disc(s, s_addn_feats)
			s_loss = criterion(s_output, real_label)
			s_loss = 100 * s_loss
			optimizer.zero_grad()
			s_loss.backward()
			optimizer.step()
			shakespeare_disc.hidden = shakespeare_disc.init_hidden()

			m = Variable(m).cuda()
			m_output = shakespeare_disc(m, m_addn_feats)
			m_loss = criterion(m_output, fake_label)
			m_loss = 100 * m_loss
			optimizer.zero_grad()
			m_loss.backward()
			optimizer.step()
			shakespeare_disc.hidden = shakespeare_disc.init_hidden()

			if idx % 100 == 0:
				print('\tepoch [{}/{}], iter: {}, s_loss: {:.4f}, m_loss: {:.4f}, preds: s: {},{}, m: {},{}'
					.format(epoch+1, num_epochs, idx, s_loss.item(), m_loss.item(), s_output.item(), np.round(s_output.item()), m_output.item(), np.round(m_output.item())))

		print('\tepoch [{}/{}]'.format(epoch+1, num_epochs))

		torch.save(shakespeare_disc.state_dict(), './shakespeare_disc.pth')

def test():
	mode = 'test'
	batch_size = 1
	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	shakespeare_disc = Discriminator(300, 256, len(dataset.vocab)).cuda()
	shakespeare_disc.load_state_dict(torch.load('./shakespeare_disc.pth'))

	shakespeare_disc.eval()

	for idx, (s, s_addn_feats, m, m_addn_feats) in tqdm(enumerate(dataloader)):
		s = s.squeeze(0)
		m = m.squeeze(0)

		s = Variable(s).cuda()
		s_output = shakespeare_disc(s, s_addn_feats)

		m = Variable(m).cuda()
		m_output = shakespeare_disc(m, m_addn_feats)

		print (idx_to_sent(s, dataset), 'Output: {}, {}'.format(s_output.item(), np.round(s_output.item())))
		print (idx_to_sent(m, dataset), 'Output: {}. {}'.format(m_output.item(), np.round(m_output.item())))

model_config = {
	'embedding_size': 300,
	'hidden_dim': 256
}
train(model_config)