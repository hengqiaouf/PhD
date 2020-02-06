from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

class FirmaData_onesubject(Dataset):
    def __init__(self, data_folder_dir, sub_id, train_part,val_part,test_part,window_size,subset='train',pre_process=True):
        matfiles = []
        assert train_part+val_part+test_part==1 #partition of three sub datasets
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
        setsize=self.datamat.shape[0]-self.window_size
        train_idx=int(setsize*train_part)
        val_idx=int(setsize*(train_part+val_part))
        if(subset=='train'):
            self.datamat=self.datamat[:train_idx]
        if(subset=='val'):
            self.datamat=self.datamat[train_idx:val_idx]
        if(subset=='test'):
            self.datamat=self.datamat[val_idx:]
    def __len__(self):
        return self.datamat.shape[0]-(self.window_size)

    def __getitem__(self, idx):
        
        return self.datamat[idx:idx+(self.window_size), :]

class FirmaData_select_subjects(Dataset):
    def __init__(self, data_folder_dir, window_size,train_part,val_part,test_part,subjects_list,subset='train',pre_process=True): #subset = train, test, val
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
        cur_label=0
        for sub_id in subjects_list:
            file_dir = os.path.join(data_folder_dir, matfiles[sub_id-1])
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
                label_temp.append(cur_label)
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
            cur_label=cur_label+1
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx): # sample size: seq_len,data_dim: [60,564]
        return self.data[idx],self.label[idx]

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
#            val_idx=int(val_part*matsize)
#            train_idx=int((train_part+val_part)*matsize)
#            if self.subset=='train':
#                self.data=self.data+data_temp[val_idx:train_idx]
#                self.label=self.label+label_temp[val_idx:train_idx]
#            if self.subset=='val':
#                self.data=self.data+data_temp[:val_idx]
#                self.label=self.label+label_temp[:val_idx]
#            if self.subset=='test':
#                self.data=self.data+data_temp[train_idx:]
#                self.label=self.label+label_temp[train_idx:]
          
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # sample size: seq_len,data_dim: [60,564]
        return self.data[idx],self.label[idx]

#def dataloader(data_folder_dir,window_size):
    

if __name__ == '__main__':
    cur_dir = os.getcwd()
    data_folder_dir = os.path.join(cur_dir, "../../data")
    dataset = FirmaData_onesubject(data_folder_dir, 1,0.7,0.0,0.3,60,subset='train')
    print(dataset.__len__())
