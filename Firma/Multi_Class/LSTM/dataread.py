from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob


class FirmaData_onesubject(Dataset):
    def __init__(self, data_folder_dir, sub_id, window_size,pre_process=True):
        matfiles = []
        for f in os.listdir(data_folder_dir):
            if f.endswith(".npy"):
                matfiles.append(f)
        matfiles.sort()
        file_dir = os.path.join(data_folder_dir, matfiles[sub_id])
        self.datamat = np.load(file_dir)
        self.window_size=window_size
        if(pre_process):
            # remove first dimension
            self.datamat = self.datamat[:, 1:]
            # binarilize
            self.datamat = np.float64(self.datamat > 0)

    def __len__(self):
        return self.datamat.shape[0]-(self.window_size)

    def __getitem__(self, idx):
        
        return self.datamat[idx:idx+(self.window_size), :]


class FirmaData_all_subjects(Dataset):
    def __init__(self, data_folder_dir, window_size,train_part,val_part,test_part,subset='train',pre_process=True): #subset = train, test, val
        matfiles = []
        assert train_part+val_part+test_part==1 #partition of three sub datasets
        self.window_size=window_size
        for f in os.listdir(data_folder_dir):
            if f.endswith(".npy"):
                matfiles.append(f)
        matfiles.sort()
        self.subset=subset
        self.data=[]
        self.label=[]
        for sub_id in range(17):
            file_dir = os.path.join(data_folder_dir, matfiles[sub_id])
            self.datamat=np.load(file_dir)
            if pre_process== True:
                # remove first dimension
                self.datamat = self.datamat[:, 1:]
                # binarilize
                self.datamat = np.float64(self.datamat > 0)
            data_temp=[]
            label_temp=[]
            for idx in range( self.datamat.shape[0]-(self.window_size)):
                data_temp.append(self.datamat[idx:idx+(self.window_size), :])
                label_temp.append(sub_id)
            matsize=len(data_temp)
            train_idx=int(train_part*matsize)
            val_idx=int((train_part+val_part)*matsize)
            if self.subset=='train':
                self.data=self.data+data_temp[:train_idx]
                self.label=self.label+label_temp[:train_idx]
            if self.subset=='val':
                self.data=self.data+data_temp[train_idx:val_idx]
                self.label=self.label+label_temp[train_idx:val_idx]
            if self.subset=='test':
                self.data=self.data+data_temp[val_idx:]
                self.label=self.label+label_temp[val_idx:]
    def __len__(self):
#        return len(self.data)
        return len(self.data)
#        return self.datamat.shape[0]-(self.window_size)

    def __getitem__(self, idx): # sample size: seq_len,data_dim: [60,564]
        return self.data[idx],self.label[idx]

#def dataloader(data_folder_dir,window_size):
    

if __name__ == '__main__':
    cur_dir = os.getcwd()
    data_folder_dir = os.path.join(cur_dir, "../../data")
    dataset_train = FirmaData_all_subjects(data_folder_dir, 60,0.5,0.2,0.3,subset='train')
    print(dataset_train.__len__())
    dataset_val = FirmaData_all_subjects(data_folder_dir, 60,0.5,0.2,0.3,subset='val')
    print(dataset_val.__len__())
    dataset_test = FirmaData_all_subjects(data_folder_dir, 60,0.5,0.2,0.3,subset='test')
    print(dataset_test.__len__())
    dataloader
