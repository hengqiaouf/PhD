import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, embed_dim,num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim) # input is not one-hot, we use fc layer as embedding
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.encoder = nn.GRU(
            self.embed_dim, self.latent_dim, self.num_layers, batch_first=True)
    def forward(self,encode_input):
        embedded = self.embedding(encode_input)
        _, last_hidden = self.encoder(embedded)
        return last_hidden
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, embed_dim,num_layers): #output_dim should be the same with input_dim
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.embed_dim=embed_dim
        self.embedding = nn.Linear(input_dim,embed_dim)
        self.output_dim = output_dim
        self.decoder = nn.GRU(self.embed_dim, self.latent_dim, self.num_layers, batch_first=True)
        #self.decode_out = nn.ReLU(nn.Linear(latent_dim, output_dim))
        self.decode_out = nn.Linear(latent_dim, output_dim)
        #self.decode_out_relu=nn.ReLU( self.decode_out)
    def forward(self,decode_input,incode_hidden):
        decode_input=decode_input.unsqueeze(1)# from [batch_size,input_size] to [batch_size,1,input_size]
        embeded= self.embedding(decode_input)
        out,last_hidden=self.decoder(embeded,incode_hidden) # when using nn.LSTM, the decode_input should be [batch_size,seq_len,data_dim] but here we decode one step at a time, so seq_len should always be 1.
#out should have shape [batch_size,seq_len,latent_dim*n directions] but here seq_len=1
        seq_gen=self.decode_out(out.squeeze(1)) # we use batch_first so the seq_len dimension is dimension 1, so we use out.squeeze(1) to make the output [batch_size,latent_dim], and the seq_gen [batch_size,output_dim]

        #seq_gen=self.decode_out_relu(seq_gen)
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
        input = torch.zeros([batch_size,data_size],device=self.device)
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

