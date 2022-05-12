
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

import os.path
import numpy as np
import torch
import open3d as o3d
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors


object_names=["/chair/","/toilet/","/table/","/airplane/","/dresser/","/bed/","/sofa/","/desk/"]

def normalize_pc(pc_np):
    ### Complete for task 1
    # Normalize the cloud to unit cube
    # input numpy ndarray -> output numpy ndarray

    # Different approaches exist. I think they are also efficient ways to preprocess the data, but they are not normalizing to unit cube.
    ## Approach 1: Normalize to unit cube
    # Max Manhattan distance should be 1.
    # manhattan_dists = np.sum(pc_np,axis=1)
    # max_dist = manhattan_dists.max() if manhattan_dists.max() > (-manhattan_dists.min()) else (-manhattan_dists.min())
    # pc_np_norm = pc_np / max_dist

    ## Approach 2: Normalize to unit sphere
    # pc_np -= np.mean(pc_np,axis=0)
    # max_norm = 0.0
    # for i in range(pc_np.shape[0]):
    #     norm = np.linalg.norm(pc_np[i,:])
    #     if norm > max_norm:
    #         max_norm = norm
    # pc_np_norm = pc_np / max_norm

    ## Approach 3: Just normalize by max dist
    pc_np -= np.mean(pc_np,axis=0)
    max_dist = pc_np.max() if pc_np.max() > (-pc_np.min()) else (-pc_np.min())
    pc_np_norm = pc_np / max_dist
    return

def voxelize(points, voxel_size=(28, 28, 28), padding_size=(32, 32, 32), resolution=0.1):
    # Voxelization implementation from VoxNet. Faster.
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.
    Args:
    `points`: pointcloud in 3D numpy.ndarray
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters
    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """

    if abs(resolution) <= 0.0:
        print('error input, resolution should not be zero')
        return None, None

    # filter outside voxel_box by using passthrough filter
    # TODO Origin, better use centroid?
    origin = (np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    points[:, 2] -= origin[2]
    # logical condition index
    x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution), (points[:, 1] >= 0))
    z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution), (points[:, 2] >= 0))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    inside_box_points = points[xyz_logical]

    # init voxel grid with zero padding_to_size=(32*32*32) and set the occupancy grid
    voxels = np.zeros(padding_size)
    # centerlize to padding box
    center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2
    # TODO currently just use the binary hit grid
    x_idx = (center_points[:, 0] / resolution).astype(int)
    y_idx = (center_points[:, 1] / resolution).astype(int)
    z_idx = (center_points[:, 2] / resolution).astype(int)
    voxels[x_idx, y_idx, z_idx] = 1

    return voxels

def spectify(points):
    ### Complete for task 3
    # Find the spectral embedding of the cloud
    # input numpy ndarray -> output numpy ndarray

    return emb

def group_knn(points,seed_points,group_size):
    # Builds a list of point groups around seed_points, can optionally be used
    neigh = NearestNeighbors(n_neighbors=group_size)
    neigh.fit(points)
    groups=neigh.kneighbors( seed_points, return_distance=False)
    groups_list=[points[groups[indexes,:],:] for indexes in range(seed_points.shape[0])]
    return groups_list

def vis_voxel_o3d(voxel_o3d):
    # Visualizes the open3d voxel object
    o3d.visualization.draw_geometries([voxel_o3d],mesh_show_wireframe=True)

def vis_points(points):
    # Visualizes the open3d point cloud object or a numpy containing point coordinates
    if type(points)==o3d.geometry.PointCloud:
        o3d.visualization.draw_geometries([points],mesh_show_wireframe=True)
    elif type(points)==np.ndarray:
        points_o3d=o3d.geometry.PointCloud()
        points_o3d.points=o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([points_o3d],mesh_show_wireframe=True)


def mesh_parser(mesh,parse_to):
    # Parses the query mesh into different representations
    if parse_to=='point':
        pc_o3d=o3d.geometry.PointCloud()
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        pc_np=np.asarray(pc_o3d.points)
        # vis_points(pc_np)
        pc_np=normalize_pc(pc_np)
        # vis_points(pc_np)
        return torch.from_numpy(pc_np).to(dtype=torch.float)
    elif parse_to=='voxel':
        ### Complete for task 2
        pc_o3d=o3d.geometry.PointCloud()
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        vox_np = normalize_pc(np.asarray(pc_o3d.points))
        vox_np = voxelize(vox_np)
        return torch.from_numpy(vox_np).to(dtype=torch.float)
    elif parse_to=='spectral':
        pc_o3d=o3d.geometry.PointCloud()
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        ### Complete for task 3
        
        spect_np=spectify(pc_np)
        return #

    


class ChallengeDataset(Dataset):

    def __init__(self, folderPath,train_test,rep_type,processed_root='processed/'):
        self.root=folderPath
        self.mode=train_test
        self.rep_type=rep_type
        self.file_list=[]
        self.label_list=[]
        self.paths=[folderPath+obj+train_test for obj in object_names]
        self.rep_type=rep_type
        self.num_classes=2
        self.processed_path=processed_root+self.rep_type+'_'+train_test
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)
        
        self.read_raw=False if os.path.exists(self.processed_path+'/0.pt') else True

        
        for label_id,path in enumerate(self.paths):
            for (dirpath, dirnames, filenames) in os.walk(path):
                self.label_list.extend([label_id for f in filenames])
                self.file_list.extend([dirpath+'/'+f for f in filenames])

    def __getitem__(self, index):
        if self.read_raw:
            mesh_o3d=o3d.io.read_triangle_mesh(self.file_list[index])
            input_tensor=mesh_parser(mesh_o3d,self.rep_type)
            target=torch.tensor(self.label_list[index],dtype=torch.int64)
            torch.save({"input": input_tensor, "target":target},self.processed_path+'/%d.pt'%index)
            return input_tensor,target
        else:
            data=torch.load(self.processed_path+'/%d.pt'%index)
            return data["input"],data["target"] 


    def __len__(self):
        return len(self.file_list)


def test_pcd(mesh_o3d):
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_o3d,number_of_points=1024)
    pc_o3d.points  = normalize_pc(np.asarray(pc_o3d.points))
    #vis_points(pc_np)

    # TODO: advanced test
    # create 3D bounding box with edge length 1. plot it together with pcd to see if everything fits inside.

    vis_points(pc_np)

def test_voxel(mesh_o3d):
    # Creating from mesh takes forever. create from pcd
    #vox_o3d=o3d.geometry.VoxelGrid().create_from_triangle_mesh(mesh_o3d, 1)
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh_o3d,number_of_points=4096)
    # For the dresser, poisson disk sampling yields more complete voxelizations.
    # But it is slower and the gaps in the voxelized geometries serve as regularization, so I stick to
    # uniform point sampling.
    #pc_o3d=o3d.geometry.TriangleMesh.sample_points_poisson_disk(mesh_o3d,number_of_points=4096)
    pc_o3d.points  = o3d.utility.Vector3dVector( normalize_pc(np.asarray(pc_o3d.points)) )

    vox_o3d=o3d.geometry.VoxelGrid().create_from_point_cloud(pc_o3d,0.05)
    vis_voxel_o3d(vox_o3d)

ex_from_diff_class = {'airplane' : 'airplane_0628.off',
                       'bed' : 'bed_0520.off',
                       'chair' : 'chair_0893.off',
                       'desk' : 'desk_0206.off',
                       'dresser' : 'dresser_0201.off',
                       'sofa' : 'sofa_0688.off',
                       'table' : 'table_0400.off',
                       'toilet' : 'toilet_0345.off'
                       }

ex_from_same_class = {'airplane' : 'airplane_0628.off',
                       'airplane2' : 'airplane_0647.off',
                       'bed' : 'bed_0520.off',
                       'bed2' : 'bed_0543.off',
                       'chair' : 'chair_0893.off',
                       'chair2': 'chair_0911.off',
                       'desk' : 'desk_0206.off',
                       'dresser' : 'dresser_0201.off',
                       'sofa' : 'sofa_0688.off',
                       'table' : 'table_0400.off',
                       'toilet' : 'toilet_0345.off'
                       }


if __name__ == "__main__":
    # mesh_o3d=o3d.io.read_triangle_mesh(self.file_list[index])

    # path = os.path.join(os.getcwd(),'dataset','airplane','test','airplane_0628.off')
    # mesh_o3d=o3d.io.read_triangle_mesh(path)
    # test_voxel(mesh_o3d)

    ## Visualize a bunch of stuff
    for key,value in ex_from_diff_class.items():
        path = os.path.join(os.getcwd(),'dataset',key,'test',value)
        mesh_o3d=o3d.io.read_triangle_mesh(path)
        test_voxel(mesh_o3d)

    #test_pcd(mesh_o3d)


