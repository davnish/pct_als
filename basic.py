import laspy
import open3d as o3d
import h5py
import numpy as np
import torch

las = laspy.read("F:\\nischal\\p_c\\PCT_Pytorch\\data\\5085_54320.las") # Reading las file
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
pcd = o3d.geometry.PointCloud()
# # concat_data = np.vstack((data[1], data[0], data[]))
# # print(concat_data.shape)
# trying = 8
data = las.xyz[:14393344].reshape(4096, 3514 , 3)
pcd.points = o3d.utility.Vector3dVector(data[50])
# print(label[trying])
# # pcd.labels = o3d.utility.Vector3dVector(label)


# pcd = pcd.voxel_down_sample(voxel_size=0.001)
o3d.visualization.draw_geometries([pcd])

def colors():
    colors = np.zeros((11930713,3 ))
    to_what = [(0, 0,0), (0,0,255), (0,255,0), (0,10,255), (255,255,0), (0,255,255), (10,255,255), (255, 0 ,255), (110,10,0)]
    # to_what = np.array(to_what)
    to_what = np.expand_dims(to_what, axis=1)
    for i,c in enumerate(to_what):
        colors += np.expand_dims((las.classification == i), axis=1) * c
