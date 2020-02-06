import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
#import dataread
import os
from dataread import FirmaData_onesubject
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report,f1_score
from autoencoder import Encoder, Decoder,Seq2Seq

# hyper-parameters
b_size = 256
subject_id_train = 1
window_size = 2  # size of sliding window, or sequence length
subject_id_anomaly=3
latent_dim = 40  # dimension of hidden state in GRU
num_layer = 1  # number of layers of GRU
learning_rate = 0.001
embed_dim=70
Max_epoch = 10
# data set up
cur_dir = os.getcwd()
data_folder_dir = os.path.join(cur_dir, "../../data")
cuda=torch.device('cuda:0')

dataset_train = FirmaData_onesubject(data_folder_dir, subject_id_train, 0.5, 0.2, 0.3, window_size, subset='train')
dataset_test_normal=FirmaData_onesubject(data_folder_dir,subject_id_train,0.5,0.2,0.3,window_size,subset='test')
dataset_test_anomaly = FirmaData_onesubject(data_folder_dir, subject_id_anomaly, 0.7, 0.0, 0.3, window_size, subset='train')

loader_train = DataLoader(dataset_train, batch_size=b_size, shuffle=True)
loader_test_normal=DataLoader(dataset_test_normal, batch_size=b_size, shuffle=True)
loader_test_anomaly=DataLoader(dataset_test_anomaly,batch_size=b_size,shuffle=True)
#
# [window_size,data_dim]  564 in our case (remove the first timestamp dimension)
input_size = dataset_train[0].shape[1]

writer = SummaryWriter()
model_encoder=Encoder(input_size,latent_dim,embed_dim,num_layer)
model_decoder=Decoder(input_size,latent_dim,input_size,embed_dim,num_layer)
model = Seq2Seq(model_encoder,model_decoder,cuda).to(cuda)
#loss_function = nn.MSELoss(reduce=False)
#loss_function = nn.L1Loss(reduce=False)
loss_function = nn.CrossEntropyLoss(weight=torch.tensor([1.,4.]).cuda()) #
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
global_step=0


#writer.add_graph(model,loader_train)
for epoch in range(Max_epoch):
    for step, seq_data in enumerate(loader_train):
#        print(seq_data.shape)  # [32,10,564], [batch,seq_len,data_dim]
        global_step+=1
        seq_data=seq_data.cuda()
        seq_pred = model(seq_data.float())
        seq_pred_cat= torch.t(torch.stack([1-seq_pred.flatten(), seq_pred.flatten()],0))
        seq_data=seq_data.flatten().long()
        loss = loss_function (seq_pred_cat,seq_data)
        # loss = loss_function(seq_pred,seq_data.float())
        # loss = loss * (1+seq_data.float()* 20) #element-wise weight
        # loss = loss.mean()
        writer.add_scalar('Loss/train',loss,global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy())
writer.close()

model.eval()

data_example=iter(loader_train)
seq_data_example=data_example.next()
seq_data_example=seq_data_example.float().cuda()
pre_data_example=model(seq_data_example,0)
y_pred=pre_data_example.cpu().detach()#b_size,seq_len,data_size
y_pred = y_pred.view(b_size,-1).numpy()
y_pred=1*(y_pred>0.5)
y_true = seq_data_example.cpu().detach()
y_true = y_true.view(b_size,-1).numpy()
f1_batch= [f1_score(true,pred) for true,pred in zip(y_true,y_pred)]
plt.hist(f1_batch)
plt.show()
y_pred_train=y_pred.flatten()

data_example=iter(loader_test_anomaly)
seq_data_example=data_example.next()
seq_data_example=seq_data_example.float().cuda()
pre_data_example=model(seq_data_example,0)
y_pred=pre_data_example.cpu().detach()#b_size,seq_len,data_size
y_pred = y_pred.view(b_size,-1).numpy()
y_pred=1*(y_pred>0.5)
y_true = seq_data_example.cpu().detach()
y_true = y_true.view(b_size,-1).numpy()
f1_batch= [f1_score(true,pred) for true,pred in zip(y_true,y_pred)]
plt.hist(f1_batch)
plt.show()
y_pred_ano=y_pred.flatten()

data_example=iter(loader_test_normal)
seq_data_example=data_example.next()
seq_data_example=seq_data_example.float().cuda()
pre_data_example=model(seq_data_example,0)
y_pred=pre_data_example.cpu().detach()#b_size,seq_len,data_size
y_pred = y_pred.view(b_size,-1).numpy()
y_pred=1*(y_pred>0.5)
y_true = seq_data_example.cpu().detach()
y_true = y_true.view(b_size,-1).numpy()
f1_batch= [f1_score(true,pred) for true,pred in zip(y_true,y_pred)]
plt.hist(f1_batch)
plt.show()
y_pred_normal=y_pred.flatten()