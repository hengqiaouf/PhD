import torch
import sklearn
import numpy as np
from sklearn.svm import OneClassSVM
from dataread import *
import os
cur_dir = os.getcwd()
data_folder_dir = os.path.join(cur_dir, "../../data") 
train_id=0
ano_id=3
par_traindata=[0.7,0,0.3]
par_anodata=[1,0,0]
window_size=1


data_train = FirmaData_onesubject(data_folder_dir, train_id,par_traindata[0],par_traindata[1],par_traindata[2]  ,window_size,subset='train')
data_test_normal= FirmaData_onesubject(data_folder_dir, train_id,par_traindata[0],par_traindata[1],par_traindata[2],window_size,subset='test')
data_test_anomaly = FirmaData_onesubject(data_folder_dir, ano_id,par_anodata[0],par_anodata[1],par_anodata[2],window_size,subset='train')


data_train=data_train.datamat
data_test_normal=data_test_normal.datamat
data_test_anomaly=data_test_anomaly.datamat


model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1,verbose=True)
model.fit(data_train)
score_train=model.score_samples(data_train)
score_test_normal=model.score_samples(data_test_normal)
score_test_anomaly=model.score_samples(data_test_anomaly)
print(np.mean(score_train),np.mean(score_test_normal),np.mean(score_test_anomaly))
