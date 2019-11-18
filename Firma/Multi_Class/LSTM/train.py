import os
import sys
import argparse
import time
import random
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from dataread import *

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import LSTMClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path',type=str,default='./saved_model')
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='data_directory')
    parser.add_argument('--subset_par',type=list,default=[0.5,0.2,0.3],
                        help='partition of training, validation and test sets')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='maximum number of epochs')
    parser.add_argument('--input_dim', type=int, default=564,
                        help='input dimension')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for random initialisation')
    args = parser.parse_args()
    train(args)


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss

def train_model(model, optimizer, train, val, max_epochs,savepath):
    criterion = nn.NLLLoss()
    for epoch in range(max_epochs):
        model.train()
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for step, (seq_data,label) in enumerate (train):
            model.zero_grad()
            seq_data=seq_data.cuda()
            label= label.cuda()
            label_pre=model(seq_data.float())
#            label_pre=model(seq_data.cuda())

            loss_step=criterion(label_pre,label)
            loss_step.backward()
            optimizer.step()
            pred_idx = torch.max(label_pre, 1)[1]
            y_true += list(label.cpu())
            
            y_pred += list(pred_idx.cpu().data.int())
            total_loss += loss_step
            if(step % 200 ==0) :
                print('train step:',step)
                print('accuracy of current batch: ',accuracy_score(label.cpu(),pred_idx.cpu().data.int()))
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_val_set(
            model, val, criterion)
        print("Epoch: {}   Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(epoch,total_loss.data.float()/len(train), acc,val_loss, val_acc))
        torch.save({
                'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        },os.path.join(savepath,'model_'+str(epoch)+'.tar'))
    return model

def evaluate_val_set(model, dat_loader_val,criterion):
    model.eval()
    y_true = list()
    y_pred = list()
    total_loss = 0
    for step, (seq_data,label) in enumerate(dat_loader_val):
        if(step % 200 ==0):
            print('val step:',step)
#        print(seq_data.shape) [batch,seq_size,data_dim]
        seq_data=seq_data.cuda()
        label=label.cuda()
        label_pre = model(seq_data.float())
        loss_step=criterion(label_pre,label)
        total_loss+=loss_step
        y_true+=list(label.cpu())
        pred_idx=torch.max(label_pre, 1)[1]
        y_pred+=list(pred_idx.cpu().data.int())
    acc=accuracy_score(y_true,y_pred)
    return total_loss.data.float()/step, acc


def evaluate_test_set(model, test, x_to_ix, y_to_ix):
    y_true = list()
    y_pred = list()

    for batch, targets, lengths, raw_data in utils.create_dataset(test, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)

        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())

    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def train(args):

    random.seed(args.seed)
    dataset_train=FirmaData_all_subjects(args.data_dir,60,args.subset_par[0],args.subset_par[1],args.subset_par[2],subset='train')
    dataset_val=FirmaData_all_subjects(args.data_dir,60,args.subset_par[0],args.subset_par[1],args.subset_par[2],subset='val')
    dataset_test=FirmaData_all_subjects(args.data_dir,60,args.subset_par[0],args.subset_par[1],args.subset_par[2],subset='test')
    dat_loader_train=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)
    dat_loader_val=DataLoader(dataset_val,batch_size=args.batch_size,shuffle=True)
    dat_loader_test=DataLoader(dataset_test,batch_size=args.batch_size,shuffle=True)



    model = LSTMClassifier (args.input_dim,
                           args.hidden_dim, output_size=17)
  #  optimizer = optim.SGD(        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.cuda()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    model = train_model(model, optimizer, dat_loader_train, dat_loader_val,args.num_epochs,args.save_path)

#    evaluate_test_set(model, test_data, char_vocab, tag_vocab)


if __name__ == '__main__':
    main()
