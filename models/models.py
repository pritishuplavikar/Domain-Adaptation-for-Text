import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from data.constants import PAD_token

class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size=1, batch_size=1, use_gpu=True, num_addn_feat=3):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.num_addn_feat = num_addn_feat

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_addn_feat, label_size),
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

    def forward(self, sentence, addn_feats):
        embeddings = self.word_embeddings(sentence)
        input_seq = embeddings.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(input_seq, self.hidden)
        addn_feat_tensor = torch.cat((lstm_out[-1], addn_feats.cuda()), 1)
        return self.hidden2label(addn_feat_tensor)