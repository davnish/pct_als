import torch
from torch.utils.data import Dataset
# import pandas as pd
import numpy as np
import laspy

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

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train = Dales()
    print(len(train))
    # a = DataLoader(train, shuffle = True, batch_size = 8)
    # print()