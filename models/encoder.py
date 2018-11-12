import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, batch_size=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size)
        self.hidden = self.init_hidden()

    def forward(self, input):
        print ('eenc ip', input.size())
        embedded = self.embedding(input)
        embedded = embedded.view(input.size(0), self.batch_size, self.input_size)
        print ('embedded', embedded.size())
        output, self.hidden = self.gru(embedded, self.hidden)
        return output, self.hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size)).cuda()
