import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.models import Discriminator
from data.shakespeare_modern import ShakespeareModern
from data.utils import idx_to_sent
from tqdm import tqdm

train_shakespeare_path = './dataset/train.original.nltktok'
test_shakespeare_path = './dataset/test.original.nltktok'
train_modern_path = './dataset/train.modern.nltktok'
test_modern_path = './dataset/test.modern.nltktok'

def train(model_config, train_config):
	mode = 'train'

	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)
	print(dataset.domain_A_max_len)
	shakespeare_disc = Discriminator(model_config['embedding_size'], model_config['hidden_dim'], len(dataset.vocab), batch_size=train_config['batch_size']).cuda()
	shakespeare_disc.train()

	if train_config['continue_train']:
		shakespeare_disc.load_state_dict(torch.load(train_config['model_path']))

	criterion = nn.BCELoss().cuda()
	optimizer = torch.optim.Adam(shakespeare_disc.parameters(), lr=train_config['base_lr'],
								 weight_decay=1e-5)

	real_label = torch.ones((train_config['batch_size'], 1)).cuda()
	fake_label = torch.zeros((train_config['batch_size'], 1)).cuda()

	for epoch in range(train_config['num_epochs']):
		for idx, (s, s_addn_feats, m, m_addn_feats) in tqdm(enumerate(dataloader)):
			s = s.transpose(0, 1)
			m = m.transpose(0, 1)

			s = Variable(s).cuda()
                        print("Before ", s.grad)
			s_output = shakespeare_disc(s, s_addn_feats)
			s_loss = criterion(s_output, real_label)
			s_loss = 100 * s_loss
			optimizer.zero_grad()
			s_loss.backward()
			print("After ", s.grad)
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
				print('\tepoch [{}/{}], iter: {}, s_loss: {:.4f}, m_loss: {:.4f}, preds: s: {}, {}, m: {}, {}'
					.format(epoch+1, train_config['num_epochs'], idx, s_loss.item(), m_loss.item(), s_output.item(), round(s_output.item()), m_output.item(), round(m_output.item())))

		print('\tepoch [{}/{}]'.format(epoch+1, train_config['num_epochs']))

		torch.save(shakespeare_disc.state_dict(), './shakespeare_disc.pth')

def test(model_config):
	mode = 'test'
	batch_size = 1
	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	shakespeare_disc = Discriminator(model_config['embedding_size'], model_config['hidden_dim'], len(dataset.vocab)).cuda()
	shakespeare_disc.load_state_dict(torch.load('./shakespeare_disc.pth'))

	shakespeare_disc.eval()

	num_correct = 0
	total_samples = 0

	for idx, (s, s_addn_feats, m, m_addn_feats) in tqdm(enumerate(dataloader)):
		s = s.transpose(0, 1)
		m = m.transpose(0, 1)

		total_samples += 2

		s = Variable(s).cuda()
		s_output = shakespeare_disc(s, s_addn_feats)

		if round(s_output.item()) == 1.0:
			num_correct += 1

		m = Variable(m).cuda()
		m_output = shakespeare_disc(m, m_addn_feats)

		if round(m_output.item()) == 0.0:
			num_correct += 1

	print ('Accuracy: {}'.format(num_correct/total_samples))

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
train(model_config, train_config)
train_config['continue_train'] = True
train_config['base_lr'] = 0.0001
train_config['num_epochs'] = 7
train(model_config, train_config)
# test(model_config)
