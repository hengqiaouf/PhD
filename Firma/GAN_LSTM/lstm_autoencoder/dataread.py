from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob


class FirmaData(Dataset):
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


if __name__ == '__main__':
    cur_dir = os.getcwd()
    data_folder_dir = os.path.join(cur_dir, "datamat")
    dataset = FirmaData(data_folder_dir, 1,2)
    print(dataset[3])
