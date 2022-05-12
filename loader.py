
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
    return pc_np_norm

def spectify(points):
    ### Complete for task 3
    # Find the spectral embedding of the cloud
    # input numpy ndarray -> output numpy ndarray
    embedding = SpectralEmbedding(n_components=3,
                                  affinity='nearest_neighbors',
                                  n_neighbors=None,
                                  random_state=1,
                                  n_jobs=-1
                                  )
    points_transformed = embedding.fit_transform(points)
    return np.concatenate((points,points_transformed),axis=-1)

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
        pc_np = normalize_pc(np.asarray(pc_o3d.points))
        return torch.from_numpy(pc_np).to(dtype=torch.float)
    elif parse_to=='voxel':
        ### Complete for task 2
        pc_o3d=o3d.geometry.PointCloud()

        ## Sample points
        # Uniform sampling is faster
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        # For the dresser, poisson disk sampling yields more complete voxelizations.
        # But it is slower and the gaps in the voxelized geometries serve as regularization, so I stick to
        # uniform point sampling.
        #pc_o3d=o3d.geometry.TriangleMesh.sample_points_poisson_disk(mesh_o3d,number_of_points=4096)

        ## VoxNet Implementation
        # vox_np = normalize_pc(np.asarray(pc_o3d.points))
        # vox_np = voxelize(vox_np)

        # Fit to unit cube
        # pc_o3d.scale(1 / np.max(pc_o3d.get_max_bound() - pc_o3d.get_min_bound()),
        #   center=pc_o3d.get_center())
        pc_o3d.points = o3d.utility.Vector3dVector( normalize_pc( np.asarray(pc_o3d.points)) )

        voxel_size = 32
        vox_o3d=o3d.geometry.VoxelGrid().create_from_point_cloud(pc_o3d,2/32)

        vox_np = np.zeros((32,32,32),dtype=np.float)
        for voxel in vox_o3d.get_voxels():
            vox_np[voxel.grid_index] = 1.0

        return torch.from_numpy(vox_np).to(dtype=torch.float)
    elif parse_to=='spectral':
        pc_o3d=o3d.geometry.PointCloud()
        pc_o3d=o3d.geometry.TriangleMesh.sample_points_uniformly(mesh,number_of_points=1024)
        ### Complete for task 3
        spect_np=spectify(np.asarray(pc_o3d.points))
        return torch.from_numpy(spect_np).to(dtype=torch.float)

    


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
    pc_np = mesh_parser(mesh_o3d,'point').numpy()
    vis_points(pc_np)

def test_voxel(mesh_o3d):
    vox_o3d = mesh_parser(mesh_o3d,'voxel') # TODO: Change the return statement of mesh_parser to use it
    vis_voxel_o3d(vox_o3d)

def test_spectral(mesh_o3d):
    pc_np = mesh_parser(mesh_o3d,'spectral').numpy()
    vis_points(pc_np[:,:3])
    vis_points(pc_np[:,3:])

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
        #test_voxel(mesh_o3d)
        #test_pcd(mesh_o3d)
        test_spectral(mesh_o3d)

