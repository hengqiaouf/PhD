import torch
import sklearn
import numpy as np
from sklearn.svm import OneClassSVM
from dataread import *
import os
cur_dir = os.getcwd()
data_folder_dir = os.path.join(cur_dir, "../data") 
par_traindata=[0.7,0,0.3]
par_anodata=[1,0,0]
window_size=1

model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1,verbose=True)
for train_id in range(17):
    test_subject_list=[*range(17)]
    test_subject_list.remove(train_id)
    data_train = FirmaData_onesubject(data_folder_dir, train_id,par_traindata[0],par_traindata[1],par_traindata[2]  ,window_size,subset='train')
    data_test_normal= FirmaData_onesubject(data_folder_dir, train_id,par_traindata[0],par_traindata[1],par_traindata[2],window_size,subset='test')
    data_test_anomaly = FirmaData_select_subjects(data_folder_dir, window_size,par_anodata[0],par_anodata[1],par_anodata[2],test_subject_list,subset='train')
    data_train=data_train.datamat
    data_test_normal=data_test_normal.datamat
    data_test_anomaly=data_test_anomaly.datamat
    model.fit(data_train)
    score_train=model.score_samples(data_train)
    score_test_normal=model.score_samples(data_test_normal)
    score_test_anomaly=model.score_samples(data_test_anomaly)
    print(np.mean(score_train),np.mean(score_test_normal),np.mean(score_test_anomaly))
