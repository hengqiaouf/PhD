import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
#import dataread
import os
from dataread import FirmaData_onesubject
from torch.utils.data import Dataset, DataLoader

# hyper-parameters
b_size = 128
window_size = 10  # size of sliding window, or sequence length
subject_id_train = 1
subject_id_anomaly=3
latent_dim = 20  # dimension of hidden state in GRU
num_layer = 1  # number of layers of GRU
learning_rate = 0.001
Max_epoch = 10
# data set up
cur_dir = os.getcwd()
data_folder_dir = os.path.join(cur_dir, "../../data")
cuda=torch.device('cuda:0')

dataset_train = FirmaData_onesubject(data_folder_dir, subject_id_train, 0.7, 0.0, 0.3, window_size, subset='train')
dataset_test_normal=FirmaData_onesubject(data_folder_dir,subject_id_train,0.7,0.0,0.3,window_size,subset='test')
dataset_test_anomaly = FirmaData_onesubject(data_folder_dir, subject_id_anomaly, 0.7, 0.0, 0.3, window_size, subset='train')

loader_train = DataLoader(dataset_train, batch_size=b_size, shuffle=True)
loader_test_normal=DataLoader(dataset_test_normal, batch_size=b_size, shuffle=True)
loader_test_anomaly=DataLoader(dataset_test_anomaly,batch_size=b_size,shuffle=True)
#
# [window_size,data_dim]  564 in our case (remove the first timestamp dimension)
input_size = dataset_train[0].shape[1]

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
    def __init__(self, input_dim, latent_dim, output_dim, num_layers): #output_dim should be the same with input_dim
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.decoder = nn.GRU(self.input_dim, self.latent_dim, self.num_layers, batch_first=True)
        self.decode_out = nn.Linear(latent_dim, output_dim)
    def forward(self,decode_input,incode_hidden):
        decode_input=decode_input.unsqueeze(1)# from [batch_size,input_size] to [batch_size,1,input_size]
        out,last_hidden=self.decoder(decode_input,incode_hidden) # when using nn.LSTM, the decode_input should be [batch_size,seq_len,data_dim] but here we decode one step at a time, so seq_len should always be 1.
#out should have shape [batch_size,seq_len,latent_dim*n directions] but here seq_len=1
        seq_gen=self.decode_out(out.squeeze(1)) # we use batch_first so the seq_len dimension is dimension 1, so we use out.squeeze(1) to make the output [batch_size,latent_dim], and the seq_gen [batch_size,output_dim]
        return seq_gen,last_hidden # seq_gen:[batch_size, output_dim]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.latent_dim == decoder.latent_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.num_layers == decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"
    def forward(self, src, teacher_forcing_ratio=0.5):
        # src = [ batch size,src sent len,data_dim]
        # trg = [ batch size,trg sent len,data_dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = src.shape[0]
        max_len = src.shape[1]
        data_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len,batch_size, data_size).to(self.device) # we need to reshape this into [batch_size,max_len,data_size] later
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden= self.encoder(src)
        # first input to the decoder zero, having same shape as the first step of input
        input = torch.zeros([batch_size,data_size],device=cuda)
        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # if teacher forcing, use actual current source sequence step
            # if not, use predicted token
            next_true_in=src[:,t,:]
            input = next_true_in if teacher_force else output
        return outputs.reshape([batch_size,max_len,-1])
writer = SummaryWriter()
model_encoder=Encoder(input_size,latent_dim,num_layer)
model_decoder=Decoder(input_size,latent_dim,input_size,num_layer)
model = Seq2Seq(model_encoder,model_decoder,cuda).to(cuda)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
global_step=0

data_example=iter(loader_train)
seq_data_example=data_example.next()
seq_data_example=seq_data_example.float().cuda()
#writer.add_graph(model,loader_train)
for epoch in range(Max_epoch):
    for step, seq_data in enumerate(loader_train):
#        print(seq_data.shape)  # [32,10,564], [batch,seq_len,data_dim]
        global_step+=1
        seq_data=seq_data.cuda()
        seq_pred = model(seq_data.float())
        loss = loss_function(seq_pred, seq_data.float())
        writer.add_scalar('Loss/train',loss,global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())
for step,seq_data in enumerate(loader_train):
    seq_data=seq_data.cuda()
    seq_pred=model(seq_data.float())
    writer.add_histogram('histogram/train',)
writer.close()
