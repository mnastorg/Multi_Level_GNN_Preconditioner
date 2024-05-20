##########################################################################
################################ PACKAGES ################################
##########################################################################

import os 
import shutil
import argparse
import sys 

import numpy as np 
from tqdm import tqdm

import mesh_random as msh 
import extract_data as dataset

##########################################################################
##########################################################################
##########################################################################

def generate_data(args):

    full_list_A_matrix, full_list_b_matrix = [], [] 
    full_list_sol, full_list_prb_data = [], []
    full_list_tags, full_list_coordinates = [], []

    total_partitions = []
    
    path_mesh = os.path.join(args.path_dataset, "mesh")
    path_data = os.path.join(args.path_dataset, "data")

    for i in tqdm(range(args.nb_problems)):

        seed = i 

        folder_mesh = os.path.join(path_mesh, "mesh_{}".format(i))
        if os.path.exists(folder_mesh) :
            shutil.rmtree(folder_mesh)
            os.mkdir(folder_mesh)
        else : 
            os.mkdir(folder_mesh)
        
        # Generate a mesh
        mesh_dict = {   "folder_to_save" : folder_mesh,
                        "name" : "full_mesh",
                        "radius" : 1.0,
                        "nb_boundary_points" : 10,
                        "hsize" : 0.02,
                        "dirichlet_tag" : 101,
                        "seed" : i
                    }
        
        myMesh = msh.MyMesh(mesh_dict)
        myMesh.generate()

        # Dictionnary data extraction
        dict_mesh = {   "mesh_path" : os.path.join(folder_mesh, "full_mesh.xdmf"),
                        "path_save_subdomains": os.path.join(folder_mesh, "saved_subdomains"),
                        "nb_nodes" : myMesh.nb_nodes,
                        "nb_nodes_subdomain": args.nb_nodes_subdomains,
                        "path_full_mesh_hdf5" : os.path.join(folder_mesh, "full_mesh_fenics.h5"),
                        "seed" : i
                    }
        
        subdata = dataset.ExtractDataPDDM(dict_mesh)

        total_partitions.append(subdata.nb_subdomains)

        list_A_sparse_matrix, list_b_matrix, list_sol, list_prb_data, list_tags, list_coordinates = subdata.save_list_subdomains_data()
        for iter in range(len(list_b_matrix)):
            for part in range(len(list_b_matrix[iter])):
                full_list_A_matrix.append(list_A_sparse_matrix[iter][str(part)])
                full_list_b_matrix.append(list_b_matrix[iter][str(part)])
                full_list_sol.append(list_sol[iter][str(part)])
                full_list_prb_data.append(list_prb_data[iter][str(part)])
                full_list_tags.append(list_tags[iter][str(part)])
                full_list_coordinates.append(list_coordinates[iter][str(part)])

    np.save(os.path.join(path_data, "A_sparse_matrix.npy"), full_list_A_matrix, allow_pickle = True)
    np.save(os.path.join(path_data, "b_matrix.npy"), full_list_b_matrix, allow_pickle = True)
    np.save(os.path.join(path_data, "sol.npy"), full_list_sol, allow_pickle = True)
    np.save(os.path.join(path_data, "prb_data.npy"), full_list_prb_data, allow_pickle = True)
    np.save(os.path.join(path_data, "tags.npy"), full_list_tags, allow_pickle = True)
    np.save(os.path.join(path_data, "coordinates.npy"), full_list_coordinates, allow_pickle = True)

    seq_nodes = [len(i) for i in full_list_coordinates]
    mean_number_of_nodes = np.mean(seq_nodes)
    min_number_of_nodes = np.min(seq_nodes)
    max_number_of_nodes = np.max(seq_nodes)

    file_path = os.path.join(path_data, "dataset_info.csv")
    file_tags = open(file_path, "w")
    file_tags.write('################## INFO ABOUT THE DATASET  #################### \n')
    file_tags.write("Number of different meshes : " + str(sum(total_partitions)) + '\n' )
    file_tags.write("Total number of instances : " + str(len(full_list_b_matrix)) + '\n' )
    file_tags.write('\n')
    file_tags.write("Mean number of nodes : " + str(mean_number_of_nodes) + '\n')
    file_tags.write("Min number of nodes : " + str(min_number_of_nodes) + '\n')
    file_tags.write("Max number of nodes : " + str(max_number_of_nodes) + '\n')
    file_tags.write('############################################################### \n')
    file_tags.close()
    
if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dataset",           type=str, default="saved_mesh", help="Folder to store mesh files")
    parser.add_argument("--nb_problems",            type=int, default=100, help="Number of meshes to create")
    parser.add_argument("--nb_nodes_subdomains",    type=int, default=500, help="Number of nodes per subdomain")

    args = parser.parse_args()

    if os.path.exists(args.path_dataset) :
        shutil.rmtree(args.path_dataset)
        os.mkdir(args.path_dataset)
    else : 
        os.mkdir(args.path_dataset)

    path_mesh = os.path.join(args.path_dataset, "mesh")
    os.mkdir(path_mesh)

    path_data = os.path.join(args.path_dataset, "data")
    os.mkdir(path_data)

    generate_data(args)
