import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from data.constants import PAD_token, SOS_token, EOS_token, DEVICE
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, max_length, dropout_p=0.1, batch_size=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size

        # self.embedding = embedding_layer
        self.attn = nn.Linear(self.hidden_size + self.input_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        # self.hidden = self.init_hidden()

    def forward(self, input, hidden, encoder_outputs):
        # print ('in dec ip', input.size())
        # input = self.embedding(input)
        embedded = self.dropout(input.view(1, self.batch_size, -1))
        # print ('dec emb size', embedded.size())
        # attn_weights = F.softmax(self.attn(torch.cat((embedded.squeeze(0), hidden.squeeze(0)), 1)), dim=1)
        # # print ('attn_weights size', attn_weights.size())
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0).transpose(0, 1),
        #     encoder_outputs.transpose(0, 1))
        # # print ('attn_applied size', attn_applied.size())
        # combined = torch.cat((embedded.squeeze(0), attn_applied.transpose(0, 1).squeeze(0)), 1)
        # output = F.relu(self.attn_combine(combined).unsqueeze(0))
        output, hidden = self.gru(embedded, hidden)
        # print ('out', output.size())
        output = self.out(output.squeeze(0))
        return output, hidden

    # def init_hidden(self):
    #     return Variable(torch.zeros(1, self.batch_size, self.hidden_size)).cuda()
