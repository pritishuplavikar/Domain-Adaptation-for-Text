import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.seq2seq_cycle_gan import Seq2SeqCycleGAN
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from data.shakespeare_modern import ShakespeareModern
from data.utils import idx_to_sent, get_addn_feats
from tqdm import tqdm

train_shakespeare_path = './dataset/train.original.nltktok'
test_shakespeare_path = './dataset/test.original.nltktok'
train_modern_path = './dataset/train.modern.nltktok'
test_modern_path = './dataset/test.modern.nltktok'

def train(model_config, train_config):
	mode = 'train'

	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)

	model = Seq2SeqCycleGAN(model_config, train_config, dataset.vocab, dataset.max_len, mode=mode)

	for epoch in range(int(train_config['num_epochs'])):
		for idx, (real_A, A_addn_feats, real_B, B_addn_feats) in tqdm(enumerate(dataloader)):

			real_A_one_hot = model.indices_to_one_hot(real_A.squeeze(0))
			real_A = Variable(real_A_one_hot).cuda()

			real_B_one_hot = model.indices_to_one_hot(real_B.squeeze(0))
			real_B = Variable(real_B_one_hot).cuda()

			A_addn_feats = A_addn_feats.cuda()
			B_addn_feats = B_addn_feats.cuda()

			model.forward(real_A, A_addn_feats, real_B, B_addn_feats)
			model.optimize_parameters()

			if idx % 1000 == 0:
				print('\tepoch [{}/{}], iter: {}, Losses: G_AtoB: {}, G_BtoA: {}, D_A: {}, D_B: {}, cycle_A: {}, cycle_B: {}, idt_A: {}, idt_B: {}'
					.format(epoch+1, train_config['num_epochs'], idx, model.loss_G_AtoB, model.loss_G_BtoA, 
						model.loss_D_A, model.loss_D_B, model.loss_cycle_A, model.loss_cycle_B, 
						model.loss_idt_A, model.loss_idt_B))

		torch.save(model.embedding_layer.state_dict(), str(epoch+1)+'_embedding_layer.pth')
		torch.save(model.G_AtoB.state_dict(), str(epoch+1)+'_G_AtoB.pth')
		torch.save(model.G_BtoA.state_dict(), str(epoch+1)+'_G_BtoA.pth')
		torch.save(model.D_A.state_dict(), str(epoch+1)+'_D_A.pth')
		torch.save(model.D_B.state_dict(), str(epoch+1)+'_D_B.pth')

		print('\tepoch [{}/{}] complete. Models saved.'.format(epoch+1, train_config['num_epochs']))

def test(model_config, train_config):
	mode = 'test'

	dataset = ShakespeareModern(train_shakespeare_path, test_shakespeare_path, train_modern_path, test_modern_path, mode=mode)
	dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=False)

	model = Seq2SeqCycleGAN(model_config, train_config, dataset.vocab, dataset.max_len, mode=mode)

	# for epoch in range(train_config['num_epochs']):
	for idx, (real_A, A_addn_feats, real_B, B_addn_feats) in tqdm(enumerate(dataloader)):
		if idx == 5:
			break

		real_A_one_hot = model.indices_to_one_hot(real_A.squeeze(0))
		real_A = Variable(real_A_one_hot).cuda()

		real_B_one_hot = model.indices_to_one_hot(real_B.squeeze(0))
		real_B = Variable(real_B_one_hot).cuda()

		A_addn_feats = A_addn_feats.cuda()
		B_addn_feats = B_addn_feats.cuda()

		model.forward(real_A, A_addn_feats, real_B, B_addn_feats)

model_config = {
	'embedding_size': 300,
	'hidden_dim': 256
}

train_config = {
	'batch_size': 1,
	'continue_train': False,
	'which_epoch': '0',
	'base_lr': 0.0001,
	'num_epochs': '10'
}

train(model_config, train_config)

train_config['which_epoch'] = train_config['num_epochs']
test(model_config, train_config)