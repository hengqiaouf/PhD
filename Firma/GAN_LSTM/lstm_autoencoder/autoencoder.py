import torch
import torch.nn as nn
import torch.optim as optim
#import dataread
import os
from dataread import FirmaData
from torch.utils.data import Dataset, DataLoader

# hyper-parameters
b_size = 32
window_size = 10  # size of sliding window, or sequence length
subject_id = 1
latent_dim = 20  # dimension of hidden state in GRU
num_layer = 1  # number of layers of GRU
learning_rate = 0.005
Max_epoch = 3
# data set up
cur_dir = os.getcwd()
data_folder_dir = os.path.join(cur_dir, "datamat")
dataset = FirmaData(data_folder_dir, subject_id, window_size)
train_loader = DataLoader(dataset, batch_size=b_size, shuffle=True)
#
# [window_size,data_dim]  564 in our case (remove the first timestamp dimension)
input_size = dataset[0].shape[1]

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            self.input_dim, self.latent_dim, self.num_layers, batch_first=True)
    def forward(self,encode_input):
         _, last_hidden = self.encoder(encode_input)
        return last_hidden
class Decoder(nn.Module):
     def __init__(self, input_dim, latent_dim,output_dim, num_layers): #output_dim should be the same with input_dim
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.decoder = nn.GRU(
            self.input_dim, self.latent_dim, self.num_layers, batch_first=True)
        self.decode_out = nn.Linear(latent_dim, output_dim) 
    def forward(self,decode_input,incode_hidden):
        decode_input=decode_input.unsqueeze(1)# from [batch_size,input_size] to [batch_sizem,1,input_size]
        out,_=self.decoder(decode_input,incode_hidden) # when using nn.LSTM, the decode_input should be [batch_size,seq_len,data_dim] but here we decode one step at a time, so seq_len should always be 1. 
#out should have shape [batch_size,seq_len,latent_dim*n directions] but here seq_len=1
        seq_gen=self.decode_out(out.squeeze(1)) # we use batch_first so the seq_len dimension is dimension 1, so we use out.squeeze(1) to make the output [batch_size,latent_dim], and the seq_gen [batch_size,output_dim]
        return seq_gen #[batch_size, output_dim]

model = FirmaGRU(input_dim=input_size, latent_dim=latent_dim, num_layers=num_layer)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(Max_epoch):
    for step, seq_data in enumerate(train_loader):
#        print(seq_data.shape)  # [32,10,564], [batch,seq_len,data_dim]
        seq_pred = model(seq_data.float())
        loss = loss_function(seq_pred, seq_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

