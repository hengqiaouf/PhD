import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataread import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
from model import LSTMClassifier
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument('--data_dir', type=str, default='../../data/',
                        help='data_directory')
    parser.add_argument('--subset_par', type=list, default=[0.8*2/3, 0.2*2/3, 1/3],
                        help='partition of training, validation and test sets')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=80,
                        help='maximum number of epochs')
#    parser.add_argument('--input_dim', type=int, default=564,
#                        help='input dimension')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for random initialisation')
    parser.add_argument('--subjects_list',type=list, default=[1,13,6],help='which subjects to use')
    parser.add_argument('--test_all', type=int, default=0, help= 'are we testing all saved models? 0 or 1')
    parser.add_argument('--test_id',type=int, default=0, help= 'model saved on which epoch to be tested')
    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if args.running_mode == 'train':
        train(args)
    elif args.running_mode == 'test':
        test(args)
    else:
        print('provide correct running mode!')


def train_model(model, optimizer, train, val, max_epochs, savepath):
    criterion = nn.NLLLoss()
    writer=SummaryWriter(savepath)
    val_losses=[]
    val_acces=[]
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
        train_loss=total_loss.data.float()/len(train)
        val_loss, val_acc = evaluate_val_set(
            model, val, criterion)
        val_losses.append(val_loss)
        val_acces.append(val_acc)
        writer.add_scalars('train_la',{'loss':train_loss,
                                             'acc':acc},epoch)
        writer.add_scalars('val_la',{'loss':val_loss,
                                           'acc':val_acc},epoch)
        writer.add_scalars('val_normed',{'loss':val_loss,
                                         'normed_acc':5*(val_acc-1),
                                         },epoch)
        print("Epoch: {}   Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(epoch,
                                                                                            train_loss, acc, val_loss,
                                                                                            val_acc))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(savepath, 'model_' + str(epoch) + '.tar'))
    val_loss_max=max(val_losses)
    val_loss_min=min(val_losses)
    val_acc_max=max(val_acces)
    val_acc_min=min(val_acces)
    val_losses_normed=[(v_l-val_loss_min)/ (val_loss_max-val_loss_min) for v_l in val_losses]
    val_acces_normed=[(v_a-val_acc_min)/(val_acc_max-val_acc_min) for v_a in val_acces]
    max_comb=0
    test_id=0
    for epoch in range(max_epochs):
        comb=val_acces_normed[epoch]+1-val_losses_normed[epoch]
        if comb>max_comb:
            max_comb = comb
            test_id=epoch
        writer.add_scalars('normed_all',{'loss':val_losses_normed[epoch],
                                        'acc':val_acces_normed[epoch],
                                        'comb':comb},epoch)
    writer.close()
    return model,test_id


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
    f1=f1_score(y_true,y_pred,average='macro') 
    if plot_conf_mat:
        fig,ax=plt.subplots()
        im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        plt.show()
    return acc,f1, conf_mat


def train(args):
    #subject_lists=[[5,7,9],[12,9,16],[2,11,5],[17,9,6],[1,13,6]]
    #subject_lists = [[1,14,15,2,6,16,7],[3,12,4,15,9,10,2],[10,14,7,11,15,8,17],[16,10,6,5,13,8,12],[17,2,13,4,7,8,16]]
    #subject_lists=[[1,7,2,8,6,11,5,15,9,3],[4,3,10,11,15,7,16,6,14,17],[5,4,12,6,10,8,15,13,2,11],[13,4,6,3,7,12,2,10,16,5],[12,15,17,13,3,9,5,14,8,2]]
    subject_lists=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
    window_sizes=[1,5,15,30,60]
    scns=['shared_data_1','shared_data_2']
    n_weeks=['1_weeks','2_weeks','3_weeks']
    random.seed(args.seed)
    logfile=open('log.txt','w+')
    for scn in scns:
        for n_week in n_weeks:
            data_dir = os.path.join(args.data_dir, scn, n_week)
            for window_size in window_sizes:
                for subjects_list in subject_lists:
                    dataset_train = FirmaData_select_subjects(data_dir, window_size, args.subset_par[0], args.subset_par[1],
                                                           args.subset_par[2], subjects_list,subset='train', pre_process=False)
                    dataset_val = FirmaData_select_subjects(data_dir, window_size, args.subset_par[0], args.subset_par[1], args.subset_par[2],subjects_list,
                                                         subset='val', pre_process=False)
                    dataset_test=FirmaData_select_subjects(data_dir, window_size, args.subset_par[0], args.subset_par[1], args.subset_par[2],subjects_list,
                                                  subset='test', pre_process=False)
                    dat_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
                    dat_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
                    dat_loader_test = DataLoader(dataset_test,batch_size=args.batch_size, shuffle=True)
                    model = LSTMClassifier(dataset_train[0][0].shape[1], args.hidden_dim, output_size=len(subjects_list))
                    model.cuda()
                    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                    save_pa= os.path.join(args.save_path, scn, n_week, str(subjects_list),str(window_size))
                    _, test_id= train_model(model, optimizer, dat_loader_train, dat_loader_val, args.num_epochs, save_pa)
                    saved_model = os.path.join(save_pa, 'model_' + str(test_id) + '.tar')
                    checkpoint = torch.load(saved_model)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    acc, f1, _ = evaluate_test_set(model, dat_loader_test)
                    logfile.write(scn+' '+n_week+' ' + 'subjects:  '+ str(subjects_list)+ 'window_size {} model {} test_accuracy:{:5.4f}, f1_score:{:5.4f}'.format(window_size,test_id,acc,f1) +"\n")
                    logfile.flush()
    logfile.close()

def test(args):
    dataset_test = FirmaData_select_subjects(args.data_dir, 30, args.subset_par[0], args.subset_par[1], args.subset_par[2],args.subjects_list,
                                          subset='test', pre_process=False)
    dat_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    if args.test_all:
        for loadid in range(args.num_epochs):
            saved_model = os.path.join(args.save_path, 'model_' + str(loadid) + '.tar')
            checkpoint = torch.load(saved_model)

            model = LSTMClassifier(dataset_test[0][0].shape[1], args.hidden_dim, output_size=3)
            model.cuda()
            model.load_state_dict(checkpoint['model_state_dict'])
            acc,f1,_= evaluate_test_set(model, dat_loader_test)
            print('model {} test_accuracy:{:5.4f}, f1_score:{:5.4f}'.format(loadid,acc,f1))
    else:
        loadid=args.test_id
        saved_model = os.path.join(args.save_path, 'model_' + str(loadid) + '.tar')
        checkpoint = torch.load(saved_model)
        model = LSTMClassifier(dataset_test[0][0].shape[1], args.hidden_dim, output_size=3)
        model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
        acc,f1, _ = evaluate_test_set(model, dat_loader_test)
        print('model {} test_accuracy:{:5.4f}, f1_score:{:5.4f}'.format(loadid,acc,f1))
if __name__ == '__main__':
    main()
