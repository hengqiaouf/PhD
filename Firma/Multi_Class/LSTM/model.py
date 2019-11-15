import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size):

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()
        self.dropout_layer = nn.Dropout(p=0.2)
    def init_hidden(self, batch_size):
#        h,c shape: [num_layers * num_directions, batch, hidden_size]
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
               autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, input_data):#input_data: batch_first[batch,seq_len,data_dim]
        self.hidden = self.init_hidden(input_data.shape[0])
        #		outputs, (ht, ct) = self.lstm(input_data, self.hidden)
        outputs, (ht, ct) = self.lstm(input_data)
    # ht is the last hidden state of the sequences
    # ht = (1 x batch_size x hidden_dim)
    # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)
        return output
