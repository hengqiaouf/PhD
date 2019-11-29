import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataread import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from model import LSTMClassifier
import matplotlib.pyplot as plt
import numpy as np

#fix random seeds
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--running_mode', type=str, default='test', help='training or testing the model')
    parser.add_argument('--load_id', type=int, default=0, help='which model to load for test')
    parser.add_argument('--save_path', type=str, default='./saved_model')
    parser.add_argument('--data_dir', type=str, default='../../data/shared_data/1_week',
                        help='data_directory')
    parser.add_argument('--subset_par', type=list, default=[0.7-0.05, 0.05, 0.3],
                        help='partition of training, validation and test sets')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=60,
                        help='maximum number of epochs')
    parser.add_argument('--input_dim', type=int, default=564,
                        help='input dimension')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for random initialisation')
    args = parser.parse_args()
    if args.running_mode == 'train':
        train(args)
    elif args.running_mode == 'test':
        test(args)
    else:
        print('provide correct running mode!')


def train_model(model, optimizer, train, val, max_epochs, savepath):
    criterion = nn.NLLLoss()
    for epoch in range(max_epochs):
        model.train()
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for step, (seq_data, label) in enumerate(train):
            model.zero_grad()
            seq_data = seq_data.cuda()
            label = label.cuda()
            label_pre = model(seq_data.float())
            #            label_pre=model(seq_data.cuda())

            loss_step = criterion(label_pre, label)
            loss_step.backward()
            optimizer.step()
            pred_idx = torch.max(label_pre, 1)[1]
            y_true += list(label.cpu())

            y_pred += list(pred_idx.cpu().data.int())
            total_loss += loss_step
            # if(step % 200 ==0) :
            #     print('train step:',step)
            #     print('accuracy of current batch: ',accuracy_score(label.cpu(),pred_idx.cpu().data.int()))
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_val_set(
            model, val, criterion)
        print("Epoch: {}   Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(epoch,
                                                                                            total_loss.data.float() / len(
                                                                                                train), acc, val_loss,
                                                                                            val_acc))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(savepath, 'model_' + str(epoch) + '.tar'))
    return model


def evaluate_val_set(model, dat_loader_val, criterion):
    model.eval()
    y_true = list()
    y_pred = list()
    total_loss = 0
    with torch.no_grad():
        for step, (seq_data, label) in enumerate(dat_loader_val):
            # if(step % 200 ==0):
            #     print('val step:',step)
            seq_data = seq_data.cuda()
            label = label.cuda()
            label_pre = model(seq_data.float())
            loss_step = criterion(label_pre, label)
            total_loss += loss_step
            y_true += list(label.cpu())
            pred_idx = torch.max(label_pre, 1)[1]
            y_pred += list(pred_idx.cpu().data.int())
        acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float() / step, acc


def evaluate_test_set(model, dat_loader_test):
    model.eval()
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for step, (seq_data, label) in enumerate(dat_loader_test):
            seq_data = seq_data.cuda()
            label = label.cuda()
            label_pre = model(seq_data.float())
            y_true += list(label.cpu())
            pred_idx = torch.max(label_pre, 1)[1]
            y_pred += list(pred_idx.cpu().data.int())
    acc = accuracy_score(y_true, y_pred)
    plot_conf_mat=False
    conf_mat = confusion_matrix(y_true, y_pred)
    if plot_conf_mat:
        fig,ax=plt.subplots()
        im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        plt.show()
    return acc, conf_mat


def train(args):
    random.seed(args.seed)
    dataset_train = FirmaData_all_subjects(args.data_dir, 60, args.subset_par[0], args.subset_par[1],
                                           args.subset_par[2], subset='train', pre_process=False)
    dataset_val = FirmaData_all_subjects(args.data_dir, 60, args.subset_par[0], args.subset_par[1], args.subset_par[2],
                                         subset='val', pre_process=False)
    dat_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dat_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    model = LSTMClassifier(args.input_dim, args.hidden_dim, output_size=17)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_model(model, optimizer, dat_loader_train, dat_loader_val, args.num_epochs, args.save_path)


def test(args):
    dataset_test = FirmaData_all_subjects(args.data_dir, 60, args.subset_par[0], args.subset_par[1], args.subset_par[2],
                                          subset='test', pre_process=False)
    dat_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    for loadid in range(args.num_epochs):
        saved_model = os.path.join(args.save_path, 'model_' + str(loadid) + '.tar')
        checkpoint = torch.load(saved_model)
        model = LSTMClassifier(args.input_dim, args.hidden_dim, output_size=17)
        model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
        acc,_= evaluate_test_set(model, dat_loader_test)
        print('model {} test_accuracy:{:5.4f}'.format(loadid,acc))
if __name__ == '__main__':
    main()
