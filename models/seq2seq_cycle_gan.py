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
import torch.nn.functional as F

class Seq2SeqCycleGAN:
	def __init__(self, model_config, train_config, vocab, max_len, mode='train'):
		self.mode = mode

		self.model_config = model_config
		self.train_config = train_config
		
		self.vocab = vocab
		self.vocab_size = self.vocab.num_words
		self.max_len = max_len

		# self.embedding_layer = nn.Embedding(vocab_size, model_config['embedding_size'], padding_idx=PAD_token)
		self.embedding_layer = nn.Sequential(
			nn.Linear(self.vocab_size, self.model_config['embedding_size']),
			nn.Sigmoid()
			)

		self.G_AtoB = Generator(self.embedding_layer, self.model_config, self.train_config, self.vocab_size, self.max_len, mode=self.mode).cuda()
		self.G_BtoA = Generator(self.embedding_layer, self.model_config, self.train_config, self.vocab_size, self.max_len, mode=self.mode).cuda()

		if self.mode == 'train':
			self.embedding_layer.train()
			self.G_AtoB.train()
			self.G_BtoA.train()

			self.D_B = Discriminator(self.embedding_layer, self.model_config, self.train_config).cuda()
			self.D_B.train()

			self.D_A = Discriminator(self.embedding_layer, self.model_config, self.train_config).cuda()
			self.D_A.train()

			self.criterionBCE = nn.BCELoss().cuda()
			self.criterionCE = nn.CrossEntropyLoss().cuda()

			self.optimizer_G = torch.optim.Adam(itertools.chain(self.embedding_layer.parameters(), self.G_AtoB.parameters(), self.G_BtoA.parameters()),
				lr=train_config['base_lr'], betas=(0.9, 0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.embedding_layer.parameters(), self.D_A.parameters(), self.D_B.parameters()),
				lr=train_config['base_lr'], betas=(0.9, 0.999))

			self.real_label = torch.ones((train_config['batch_size'], 1)).cuda()
			self.fake_label = torch.zeros((train_config['batch_size'], 1)).cuda()
		else:
			self.embedding_layer.load_state_dict(torch.load('embedding_layer.pth'))

			self.G_AtoB.load_state_dict(torch.load('G_AtoB.pth'))
			self.G_BtoA.load_state_dict(torch.load('G_BtoA.pth'))

			self.embedding_layer.eval()
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

		self.clip_gradient(self.embedding_layer)
		self.clip_gradient(netD)

		return loss_D

	def backward_D_A(self):
		self.loss_D_A = self.backward_D_basic(self.D_A, self.real_A, self.real_A_addn_feats, self.fake_A, self.fake_A_addn_feats)

	def backward_D_B(self):
		self.loss_D_B = self.backward_D_basic(self.D_B, self.real_B, self.real_B_addn_feats, self.fake_B, self.fake_B_addn_feats)

	def backward_G(self):
		self.D_B.hidden = self.D_B.init_hidden()
		self.fake_B_addn_feats = get_addn_feats(self.fake_B, self.vocab).cuda()
		self.loss_G_AtoB = self.criterionBCE(self.D_B(self.fake_B, self.fake_B_addn_feats), self.real_label)

		self.D_A.hidden = self.D_A.init_hidden()
		self.fake_A_addn_feats = get_addn_feats(self.fake_A, self.vocab).cuda()
		self.loss_G_BtoA = self.criterionBCE(self.D_A(self.fake_A, self.fake_A_addn_feats), self.real_label)
		
		if self.rec_A.size(0) != self.real_A_label.size(0):
			self.real_A, self.rec_A, self.real_A_label = self.update_label_sizes(self.real_A, self.rec_A, self.real_A_label)
		self.loss_cycle_A = self.criterionCE(self.rec_A, self.real_A_label) #* lambda_A

		if self.rec_B.size(0) != self.real_B_label.size(0):
			self.real_B, self.rec_B, self.real_B_label = self.update_label_sizes(self.real_B, self.rec_B, self.real_B_label)
		self.loss_cycle_B = self.criterionCE(self.rec_B, self.real_B_label) #* lambda_B

		self.idt_B = self.G_AtoB(self.real_B)
		if self.idt_B.size(0) != self.real_B_label.size(0):
			self.real_B, self.idt_B, self.real_B_label = self.update_label_sizes(self.real_B, self.idt_B, self.real_B_label)
		self.loss_idt_B = self.criterionCE(self.idt_B, self.real_B_label) #* lambda_B * lambda_idt

		self.idt_A = self.G_BtoA(self.real_A)
		if self.idt_A.size(0) != self.real_A_label.size(0):
			self.real_A, self.idt_A, self.real_A_label = self.update_label_sizes(self.real_A, self.idt_A, self.real_A_label)
		self.loss_idt_A = self.criterionCE(self.idt_A, self.real_A_label) #* lambda_A * lambda_idt

		self.loss_G = self.loss_G_AtoB + self.loss_G_BtoA + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
		self.loss_G.backward()

		self.clip_gradient(self.embedding_layer)
		self.clip_gradient(self.G_AtoB)
		self.clip_gradient(self.G_BtoA)

	def forward(self, real_A, real_A_addn_feats, real_B, real_B_addn_feats):
		self.real_A = real_A
		self.real_A_addn_feats = real_A_addn_feats
		self.real_A_label = self.real_A.max(dim=1)[1]

		self.real_B = real_B
		self.real_B_addn_feats = real_B_addn_feats
		self.real_B_label = self.real_B.max(dim=1)[1]

		self.fake_B = F.softmax(self.G_AtoB.forward(self.real_A), dim=1)

		# print (self.real_A.max(dim=1)[1].tolist(), self.rec_A.max(dim=1)[1].tolist())

		self.fake_A = F.softmax(self.G_BtoA.forward(self.real_B), dim=1)
		
		# print (self.real_B.max(dim=1)[1].tolist(), self.rec_B.max(dim=1)[1].tolist())

		if self.mode == 'train':
			self.rec_A = self.G_BtoA.forward(self.fake_B)
			self.rec_B = self.G_AtoB.forward(self.fake_A)

			self.set_requires_grad([self.D_A, self.D_B], False)

			self.optimizer_G.zero_grad()
			self.backward_G()
			self.optimizer_G.step()

			self.set_requires_grad([self.D_A, self.D_B], True)
			self.optimizer_D.zero_grad()
			self.backward_D_B()
			self.backward_D_A()
			self.optimizer_D.step()
		else:
			fake_B_list = self.fake_B.max(dim=1)[1].tolist()
			fake_A_list = self.fake_A.max(dim=1)[1].tolist()

			print ('Shakespeare to Modern: ', idx_to_sent(fake_B_list, self.vocab))
			print ('Modern to Shakespeare: ', idx_to_sent(fake_A_list, self.vocab))

	def update_label_sizes(self, real, rec, real_label):
		
		if rec.size(0) > real.size(0):
			real_label = torch.cat((real_label, torch.zeros((rec.size(0)-real.size(0))).type(torch.LongTensor).cuda()), 0)
		elif rec.size(0) < real.size(0):
			diff = real.size(0)-rec.size(0)
			to_concat = torch.zeros((diff, self.vocab_size)).cuda()
			to_concat[:, 0] = 1
			rec = torch.cat((rec, to_concat), 0)

		return real, rec, real_label

	def indices_to_one_hot(self, idx_tensor):
		one_hot_tensor = torch.empty((idx_tensor.size(0), self.vocab_size))
		for idx in range(idx_tensor.size(0)):
			zeros = torch.zeros((self.vocab_size))
			zeros[idx_tensor[idx].item()] = 1.0
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
