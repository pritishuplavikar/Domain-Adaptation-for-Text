import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.generator import Generator
from models.discriminator import Discriminator
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from data.shakespeare_modern import ShakespeareModern
from data.utils import idx_to_sent, get_addn_feats
from tqdm import tqdm
import numpy as np
import itertools
from torchviz import make_dot

train_shakespeare_path = './dataset/train.original.nltktok'
test_shakespeare_path = './dataset/test.original.nltktok'
train_modern_path = './dataset/train.modern.nltktok'
test_modern_path = './dataset/test.modern.nltktok'

class Seq2SeqCycleGAN:
	def __init__(self, model_config, train_config, vocab, max_len, mode='train'):
		self.mode = mode

		self.model_config = model_config
		self.train_config = train_config
		
		self.vocab = vocab
		self.vocab_size = self.vocab.num_words
		self.max_len = max_len

		self.G_AtoB = Generator(self.model_config, self.train_config, self.vocab_size, self.max_len, mode=self.mode).cuda()
		self.G_BtoA = Generator(self.model_config, self.train_config, self.vocab_size, self.max_len, mode=self.mode).cuda()
		
		self.softmax = nn.Softmax(dim=1)

		if self.mode == 'train':
			self.G_AtoB.train()
			self.G_BtoA.train()

			self.D_B = Discriminator(self.model_config, self.train_config, self.vocab_size).cuda()
			self.D_B.train()

			self.D_A = Discriminator(self.model_config, self.train_config, self.vocab_size).cuda()
			self.D_A.train()

			self.criterionBCE = nn.BCELoss().cuda()
			self.criterionCE = nn.CrossEntropyLoss().cuda()

			self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()),
				lr=train_config['base_lr'], betas=(0.9, 0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
				lr=train_config['base_lr'], betas=(0.9, 0.999))

			self.real_label = torch.ones((train_config['batch_size'], 1)).cuda()
			self.fake_label = torch.zeros((train_config['batch_size'], 1)).cuda()
		else:
			self.G_AtoB.eval()
			self.G_BtoA.eval()

	def backward_D_basic(self, netD, real, real_addn_feats, fake, fake_addn_feats):
		netD.hidden = netD.init_hidden()
		pred_real = netD(real, real_addn_feats)
		loss_D_real = self.criterionBCE(pred_real, self.real_label)

		netD.hidden = netD.init_hidden()
		pred_fake = netD(fake.detach(), fake_addn_feats)
		loss_D_fake = self.criterionBCE(pred_fake, self.fake_label)

		loss_D = (loss_D_real + loss_D_fake) * 0.5
		loss_D.backward()

		self.clip_gradient(netD)

		return loss_D

	def backward_D_A(self):
		self.loss_D_A = self.backward_D_basic(self.D_A, self.real_A, self.real_A_addn_feats, self.fake_A_labels, self.fake_A_addn_feats) * 10

	def backward_D_B(self):
		self.loss_D_B = self.backward_D_basic(self.D_B, self.real_B, self.real_B_addn_feats, self.fake_B_labels, self.fake_B_addn_feats) * 10

	def backward_G(self):
		self.D_B.hidden = self.D_B.init_hidden()
		self.fake_B_addn_feats = get_addn_feats(self.fake_B_labels, self.vocab).cuda()
		self.loss_G_AtoB = self.criterionBCE(self.D_B(self.fake_B_labels, self.fake_B_addn_feats), self.real_label) * 10

		self.D_A.hidden = self.D_A.init_hidden()
		self.fake_A_addn_feats = get_addn_feats(self.fake_A_labels, self.vocab).cuda()
		self.loss_G_BtoA = self.criterionBCE(self.D_A(self.fake_A_labels, self.fake_A_addn_feats), self.real_label) * 10

		real_A_label = self.real_A.max(dim=1)[1]
		self.loss_cycle_A = self.criterionCE(self.rec_A, real_A_label) #* lambda_A

		real_B_label = self.real_B.max(dim=1)[1]
		self.loss_cycle_B = self.criterionCE(self.rec_B, real_B_label) #* lambda_B

		self.idt_B, _ = self.G_AtoB(self.real_B)
		self.loss_idt_B = self.criterionCE(self.idt_B, real_B_label) #* lambda_B * lambda_idt

		self.idt_A, _ = self.G_BtoA(self.real_A)
		self.loss_idt_A = self.criterionCE(self.idt_A, real_A_label) #* lambda_A * lambda_idt

		self.loss_G = self.loss_G_AtoB + self.loss_G_BtoA + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
		self.loss_G.backward()

		self.clip_gradient(self.G_AtoB)
		self.clip_gradient(self.G_BtoA)

	def forward(self, real_A, real_A_addn_feats, real_B, real_B_addn_feats):
		self.real_A = real_A
		self.real_A_addn_feats = real_A_addn_feats

		self.real_B = real_B
		self.real_B_addn_feats = real_B_addn_feats

		_, self.fake_B_labels = self.G_AtoB.forward(self.real_A)
		self.rec_A, _ = self.G_BtoA.forward(self.fake_B_labels)

		_, self.fake_A_labels = self.G_BtoA.forward(self.real_B)
		self.rec_B, _ = self.G_AtoB.forward(self.fake_A_labels)

		self.set_requires_grad([self.D_A, self.D_B], False)
		self.optimizer_G.zero_grad()
		self.backward_G()
		self.optimizer_G.step()

		self.set_requires_grad([self.D_A, self.D_B], True)
		self.optimizer_D.zero_grad()
		self.backward_D_B()
		self.backward_D_A()
		self.optimizer_D.step()

	def indices_to_one_hot(self, idx_tensor):
		one_hot_tensor = torch.empty((idx_tensor.size(0), self.vocab_size))
		for idx in range(idx_tensor.size(0)):
			zeros = torch.zeros((self.vocab_size))
			zeros[idx_tensor[idx, 0].item()] = 1.0
			one_hot_tensor[idx] = zeros

		return one_hot_tensor	

	def set_requires_grad(self, nets, requires_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def clip_gradient(self, model):
		nn.utils.clip_grad_norm_(model.parameters(), 0.25)