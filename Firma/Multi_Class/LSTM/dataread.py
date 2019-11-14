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
    def __init__(self, data_folder_dir, window_size,pre_process=True):
        matfiles = []
        self.window_size=window_size
        for f in os.listdir(data_folder_dir):
            if f.endswith(".npy"):
                matfiles.append(f)
        matfiles.sort()
        self.data=[]
        self.label=[]
        for sub_id in range(17):
            file_dir = os.path.join(data_folder_dir, matfiles[sub_id])
            self.datamat = np.load(file_dir)
            if(pre_process):
                # remove first dimension
                self.datamat = self.datamat[:, 1:]
                # binarilize
                self.datamat = np.float64(self.datamat > 0)
            for idx in range( self.datamat.shape[0]-(self.window_size)):
                self.data.append(self.datamat[idx:idx+(self.window_size), :])
                self.label.append(sub_id)
    def __len__(self):
        return len(self.data)
#        return self.datamat.shape[0]-(self.window_size)

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]
#        return self.datamat[idx:idx+(self.window_size), :],label

if __name__ == '__main__':
    cur_dir = os.getcwd()
    data_folder_dir = os.path.join(cur_dir, "../../data")
    dataset = FirmaData_all_subjects(data_folder_dir, 60)
    print(dataset.__len__())
