import torch
from torch.utils.data import Dataset
# import pandas as pd
# import glob
import os
import numpy as np
import laspy
import open3d as o3d

class Dales(Dataset):
    def __init__(self, grid_size, points_taken, partition='train'):
        self.data, self.label = load_data(grid_size, points_taken)


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

def load_data(grid_size, points_taken):
    # path = "data"
    if os.path.exists(f'data\\train_test_{grid_size}_{points_taken}.npz'): # this starts from the system's path
        tiles = np.load(f'data\\train_test_{grid_size}_{points_taken}.npz')
        tiles_np = tiles['x']
        tiles_np_labels = tiles['y']
    else:
        las = laspy.read("F:\\nischal\\p_c\\pct_als\\data\\5140_54445.las")
        las_classification = las_label_replace(las)
        # grid_size = grid_size
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

        tiles = []
        tiles_labels = []

        grid_lengths = [len(i) for i in grid_point_clouds.values()]
        min_grid_points = (max(grid_lengths) - min(grid_lengths)) * 0.1
        min_points = min(grid_lengths)

        for grid, label in zip(grid_point_clouds.values(), grid_point_clouds_label.values()):

            len_grid = len(grid)

            if(len_grid - min_points>min_grid_points): # This is for excluding points which are at the boundry at the edges of the tiles
                if(len_grid<points_taken): # This is for if the points in the grid are less then the required points for making the grid
                    for _ in range(points_taken-len_grid):
                        grid.append(grid[0])
                        label.append(label[0])
                tiles.append(grid[:points_taken])
                tiles_labels.append(label[:points_taken])

        tiles_np = np.asarray(tiles)

        tiles_np_labels = np.asarray(tiles_labels)
        np.savez(f'data\\train_test_{grid_size}_{points_taken}.npz', x = tiles_np, y = tiles_np_labels)

    return tiles_np, tiles_np_labels

def visualize():
    las_xyz, _ = load_data(10, 4096)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(las_xyz[0])
    # pcd.colors = o3d.utility.Vector3dVector(give_colors(las_xyz[0], las_label[0], partition = 'train'))
    o3d.visualization.draw_geometries([pcd])



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    visualize()
    # a = DataLoader(train, shuffle = True, batch_size = 8)
    # print()