import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from data.constants import PAD_token

class Discriminator(nn.Module):
    def __init__(self, model_config, train_config, vocab_size, use_gpu=True, num_addn_feat=3):
        super(Discriminator, self).__init__()
        self.hidden_dim = model_config['hidden_dim']
        self.batch_size = train_config['batch_size']
        self.use_gpu = use_gpu
        self.num_addn_feat = num_addn_feat
        self.vocab_size = vocab_size

        self.embedding_layer = nn.Embedding(self.vocab_size, model_config['embedding_size'], padding_idx=PAD_token)
        self.lstm = nn.LSTM(model_config['embedding_size'], self.hidden_dim)
        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_addn_feat, 1), 
            nn.Sigmoid()
            )
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, input, addn_feats):
        embeddings = self.embedding_layer(input).view(input.size(0), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        addn_feat_tensor = torch.cat((lstm_out[-1], addn_feats), 1)
        return self.hidden2label(addn_feat_tensor)