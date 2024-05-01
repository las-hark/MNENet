import os
from matplotlib import pyplot as plt
import numpy as np
import h5py
from sklearn import preprocessing

def get_datasetname(file_name_with_dir : str) -> str:
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split( "_" )[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def get_data(folder_path : str) -> [np.array,np.array]:
   #initialize
    h5_files = [file for file in os.listdir(folder_path) if file.endswith('.h5')]
    np_dataset = np.zeros([len(h5_files), 248, 35624]) 
    np_label   = np.array([1 if s.startswith('r') else 2 if s.startswith('task_m') else 3 if s.startswith('task_s') else 4 for s in h5_files])
    i = 0

    for filename in h5_files:
        filename_path = os.path.join(folder_path, filename)
        with h5py.File(filename_path, 'r') as f:
            #read data
            datasetname = get_datasetname(filename_path)
            matrix = f.get(datasetname)[()]
            # matrix contains the data from the current H5 file
            #print(f"File: {filename}, Type: {type(matrix)}, Shape: {matrix.shape}")
            np_dataset[i] = matrix
            i += 1

    return np_dataset, np_label


# z_score normalize data
def z_nor(dataset : np.array) -> np.array:
    dataset_nor = np.array([preprocessing.scale(data) for data in dataset])
    return dataset_nor
    
folder_path = r'FinalProjectdata/Cross/train/' 
dataset, label = get_data(folder_path)
dataset_nor = z_nor(dataset)
np.save('cross_train_set_nor.npy',dataset_nor)
np.save('cross_train_set.npy',dataset)
np.save('cross_train_label.npy',label)

## read h5, then save the dataset and labelset as npy
folder_path_intratrain = r'FinalProjectdata/Intra/train/' 
intra_train_set, intra_train_label = get_data(folder_path_intratrain)
np.save('intra_train_set.npy',intra_train_set)
np.save('intra_train_label.npy',intra_train_label)

folder_path_intratest = r'FinalProjectdata/Intra/test/' 
intra_test_set, intra_test_label = get_data(folder_path_intratest)
np.save('intra_test_set.npy',intra_test_set)
np.save('intra_test_label.npy',intra_test_label)

## normalize data and save
intra_train_set_nor = z_nor(intra_train_set)
intra_test_set_nor  = z_nor(intra_test_set)
np.save('intra_train_set_nor.npy',intra_train_set_nor)
np.save('intra_test_set_nor.npy',intra_test_set_nor)

## sample drawing
plt.imshow(np.reshape(intra_train_set_nor[0,:,0],(8,31)))
plt.colorbar()