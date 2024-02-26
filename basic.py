import laspy
import open3d as o3d
import h5py
import numpy as np
import torch


# las = laspy.read("F:\\nischal\\p_c\pct_als\\data\\5140_54445.las") # Reading las file


def griding():
    grid_size = 20
    grid_point_clouds = {}
    grid_point_clouds_label = {}
    for point, label in zip(las.xyz, las.classification):
        grid_x = int(point[0] / grid_size)
        grid_y = int(point[1] / grid_size)

        if (grid_x, grid_y) not in grid_point_clouds:
            grid_point_clouds[(grid_x, grid_y)] = []
            grid_point_clouds_label[(grid_x, grid_y)] = []
        
        grid_point_clouds[(grid_x, grid_y)].append(point)
        grid_point_clouds_label[(grid_x, grid_y)].append(label)

    print(len(grid_point_clouds_label))
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


# print(las_xyz.shape)

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


    return colors.reshape(-1, 3)

torch.manual_seed(42)
a = torch.randn(8, 4096, 4)
# print(a)
print(a.max(dim = -1)[1].shape)


# las.classification

# cnt  = 0
# for i in grid_point_clouds.keys():
#     cnt += 1

def visualize():
    las_xyz, las_label = griding()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(las_xyz[10])
    pcd.colors = o3d.utility.Vector3dVector(give_colors(las_xyz[10], las_label[10], partition = 'train'))
    o3d.visualization.draw_geometries([pcd])

# data = [[] for _ in range(400)]
# cnt_grid = 0

# print(data.shape)

# data = np.vstack([value for value in grid_point_clouds.values()])

# print(data.shape)
        

    






# print(las.xyz.reshape((2331, ,3)))

# f = h5py.File("F:\\nischal\\p_c\\PCT_Pytorch\\data\\modelnet40_ply_hdf5_2048\\ply_data_train0.h5")
# data = f['data'][0]
# label = f['label'][0].astype('int64')
# with h5py.File('F:\\nischal\\p_c\\PCT_Pytorch\\data\\modelnet40_ply_hdf5_2048\\ply_data_train0.h5', 'r') as f:
#     for key in f.keys():
#         print(key)
#     data = f['data'][()]
#     label = f['label'][()]

# print(np.unique(label))
# # print(data.reshape(2048*2048,3))
# # print(las.xyz[3][:2])
# # print(data.shape)
# # print(label.shape)
# # data = data.reshape(2048*2048,3)

# # concat_data = np.vstack((data[1], data[0], data[]))
# # print(concat_data.shape)
# trying = 8

# data = las.xyz[:12246290].reshape(3485, 3514 , 3)

# print(len(data))

# print(label[trying])
# pcd.labels = o3d.utility.Vector3dVector(label)



# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(las.xyz)
# downpcd = pcd.voxel_down_sample(voxel_size=2)
# downpcd.points = o3d.utility.Vector3dVector(np.asarray(downpcd.points)[:167936].reshape(82, 2048 , 3)[0])

# print(las.xyz.shape)

# pcd.points = o3d.utility.Vector3dVector()
# pcd.colors = o3d.utility.Vector3dVector(give_colors(pcd, to_see=0, partition = 'test'))

# print(np.array(pcd.points).shape)

# o3d.visualization.draw_geometries([pcd])

# data = np.array(pcd.points)[:167936].reshape(82, 2048 , 3)
# print(data[0].shape)
# pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points))

# print(give_colors(pcd, to_see=1))

# print(np.asarray(pcd.colors))