import numpy as np
from torch.utils.data import Dataset
import laspy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split



class Dales(Dataset):
    def __init__(self, partition='train'):
        self.data, self.label = load_data(partition)


    def __getitem__(self, item):
        pointcloud = torch.tensor(self.data[item]).float()
        label = torch.tensor(self.label[item])

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def las_label_replace(las):
    las_classification = np.asarray(las.classification)
    mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}
    for old, new in mapping.items():
        las_classification[las_classification == old] = new
    return las_classification

def load_data(partition):
    las = laspy.read("F:\\nischal\\p_c\pct_als\\data\\5140_54445.las")
    las_classification = las_label_replace(las)
    grid_size = 20
    grid_point_clouds = {}
    grid_point_clouds_label = {}
    for point, label in zip(las.xyz, las_classification):
        grid_x = int(point[0] / grid_size)
        grid_y = int(point[1] / grid_size)

        if (grid_x, grid_y) not in grid_point_clouds:
            grid_point_clouds[(grid_x, grid_y)] = []
            grid_point_clouds_label[(grid_x, grid_y)] = []
        
        grid_point_clouds[(grid_x, grid_y)].append(point)
        grid_point_clouds_label[(grid_x, grid_y)].append(label)

    # print(len(grid_point_clouds_label))
    tiles = []
    tiles_labels = []
    for i, j in zip(grid_point_clouds.values(), grid_point_clouds_label.values()):
        grid = i
        len_grid = len(grid)
        label = j
        if(len_grid>100):
            if(len_grid<4096):
                for _ in range(4096-len_grid):
                    grid.append(grid[0])
                    label.append(label[0])
            tiles.append(grid[:4096])
            tiles_labels.append(label[:4096])

    tiles_np = np.asarray(tiles)
    # tiles_np_labels = np.expand_dims(np.asarray(tiles_labels), axis = 2)
    tiles_np_labels = np.asarray(tiles_labels)
    return tiles_np, tiles_np_labels

def give_colors(las_xyz, las_label ,to_see = None, partition = 'test'):
    to_what = [(0,255,255), (0,0,255), (0,255,0), (0,10,255), (255,255,0), (0,255,255), (10,255,255), (255, 0 ,255), (110,10,0)]
    colors = np.zeros(las_xyz.shape)
    if partition == 'train':
        # to_what = np.array(to_what)
        to_what = np.expand_dims(to_what, axis=1)
        for i,c in enumerate(to_what):
            colors += np.expand_dims((las_label == i), axis=1) * c
    else:      
        # To see the pcd data into visulization first we need to convert it into the a one shape array.  
        # to_what = [(0,0,255), (0,255,0)]
        colors[:,:] = to_what[0]
        colors[0, :] = to_what[1]

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    print ('use random drop', len(drop_idx))

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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import time
    train = Dales()
    # print(train[10])
    # print(len(train) *0.9)
    # train_dataset, test_dataset = random_split(train, [0.9, 0.1])
    data_loader = DataLoader(train, shuffle=True, num_workers=1, batch_size=64)
    
    start = time.time()
    for data, label in data_loader:
        st = time.time()
        data, label = data.to('cuda'), label.to('cuda')
        ed = time.time()
        print(ed-st)
    end = time.time()

    print(f'Total_time: {end-start}')
