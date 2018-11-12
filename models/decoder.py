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

        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.attn = nn.Linear(self.hidden_size + self.input_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + self.input_size, self.input_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.hidden = self.init_hidden()

    def forward(self, input, encoder_outputs):
        print ('dec ip', input.size())
        embedded = self.embedding(input).view(self.batch_size, self.input_size)
        print ('embedded', embedded.size())
        embedded = self.dropout(embedded)
        attn_int = self.attn(torch.cat((embedded[0], self.hidden[0]), 1))
        print (attn_int.size())
        attn_weights = F.softmax(attn_int, dim=1)
        print ('attn_weights', attn_weights.size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        print ('attn_applied', attn_applied.size())
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        print ('attn_combine', attn_combine.size())
        output = F.relu(output)
        output, self.hidden = self.gru(output, self.hidden)
        print ('gru output', output.size())
        output = F.log_softmax(self.out(output[0]), dim=1)
        print ('final output', output.size())
        return output, self.hidden, attn_weights

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size)).cuda()
