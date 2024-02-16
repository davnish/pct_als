import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import laspy
import torch



# def load_data(partition):
#     # download()
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, 'data')
#     all_data = []
#     all_label = []
#     for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
#         f = h5py.File(h5_name)
#         data = f['data'].astype('float32')
#         label = f['label'].astype('int64')
#         # print(label)
#         f.close()
#         all_data.append(data)
#         all_label.append(label)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     # print(all_data)
#     return all_data, all_label


def load_data(partition):
    las = laspy.read("F:\\nischal\\p_c\\PCT_Pytorch\\data\\5085_54320.las")
    # print(np.array(las.xyz))
    return las.xyz[:14393344].reshape((4096, 3514 , 3)), np.array(las.classification)[:14393344].reshape((4096, 3514 , 1))

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class Dales(Dataset):
    def __init__(self, partition='train'):
        self.data, self.label = load_data(partition)
        self.partition = partition     
        # self.data =    

    def __getitem__(self, item):
        pointcloud = torch.tensor(self.data[item]).float()
        label = torch.tensor(self.label[item])

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

from torch.utils.data import random_split

if __name__ == '__main__':
    train = Dales()
    # print(train[10])
    # print(len(train) *0.9)
    # train_dataset, test_dataset = random_split(train, [0.9, 0.1])
    # test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.dtype)
        print(label.shape)
        break
