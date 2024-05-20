###############################
########## Packages ###########
###############################
 
import warnings
warnings.filterwarnings('ignore')

import os 
import pickle

import argparse 

import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy
import torch 
import json 
import time 
import re
import mesh_random as msh 
import partitioning_original as pt
import resolution_optimized as solve
import model_dss_inference as modtest

###############################
###############################
###############################

n_samples = 2

radius = 1.2
size_subdomains = 1000
overlap = 2
path_savings = "results/test_saved_10k_multiple/"
path_models = "all_results"

if not os.path.exists(path_savings):
    os.mkdir(path_savings)

tolerance = 1.e-6
max_iter = 500
level = 2
###############################
###############################
###############################

def custom_sort(item):
    # Extract numbers after 'k' and 'd'
    k_number = int(re.search(r'k(\d+)', item).group(1))
    d_number = int(re.search(r'd(\d+)', item).group(1))
    return k_number, d_number


sorted_data = sorted(os.listdir(path_models), key=custom_sort)
list_path = [os.path.join("all_results", i, "ckpt", "new_model.pt") for i in sorted_data]
list_names = sorted_data

saved_dict = {}

for n in range(n_samples):
    
    seed = 1.e2*(n+1)
    
    saved_dict[str(n)] = {}

    print("Step 1 ---> Mesh construction")
    mesh_dict = {   "folder_to_save" : os.path.join(path_savings, "mesh"),
                    "name" : "temp",
                    "radius" : radius,
                    "nb_boundary_points" : 10,
                    "hsize" : 0.02,
                    "dirichlet_tag" : 101,
                    "seed" : seed
                }

    myMesh = msh.MyMesh(mesh_dict)
    myMesh.generate()
    print(myMesh.nb_nodes)

    print("Step 2 ---> Partitioning")
    part_dict = {   "path_mesh_xdmf" : os.path.join(path_savings, "mesh", "temp.xdmf"),
                    "path_save_subdomains" : os.path.join(path_savings, "subdomains"),
                    "Nparts" : int(np.ceil(myMesh.nb_nodes / size_subdomains)),
                    "Noverlaps" : overlap
                }

    partition = pt.Partitioning(part_dict)
    partition.framework_partitioning()

    Nsub = int(np.mean([len(partition.dict_restriction[str(i)]) for i in range(part_dict["Nparts"])]))
    print('Number of nodes per subdomains : ', Nsub)

    print("-----> GPU WARMUP <-----")
    dict_resolution = { "partition" : partition,
                        "path_full_mesh_hdf5" : os.path.join(path_savings, "mesh", "temp_fenics.h5"),
                        "path_subdomains" : os.path.join(path_savings, "subdomains"),
                        "path_model" : list_path[0],
                        "radius" : radius,
                        "seed" : seed,
                        "path_savings": path_savings
                        }
    poisson = solve.DDMResolution(dict_resolution)
    sol, res, infer_time = poisson.solve_cg(    preconditioner = "ASM-GNN",
                                                level = level,
                                                stop_mode = "res_rel",
                                                x0 = None,
                                                tol = tolerance,
                                                max_iter = 5)
    print("First inference time : ", infer_time)

    print("-----> START <-----")

    for i in range(len(list_path)):
        
        saved_dict[str(n)][list_names[i]] = {}

        print(f"Solve for model {list_names[i]}")

        dict_resolution = { "partition" : partition,
                            "path_full_mesh_hdf5" : os.path.join(path_savings, "mesh", "temp_fenics.h5"),
                            "path_subdomains" : os.path.join(path_savings, "subdomains"),
                            "path_model" : list_path[i],
                            "radius" : radius,
                            "seed" : n,
                            "path_savings": path_savings
                            }
        poisson = solve.DDMResolution(dict_resolution)

        start = time.time()
        sol, res, infer_time = poisson.solve_cg(    preconditioner = "ASM-GNN",
                                                    level = level,
                                                    stop_mode = "res_rel",
                                                    x0 = None,
                                                    tol = tolerance,
                                                    max_iter = max_iter)
        end = time.time()

        saved_dict[str(n)][list_names[i]]["mean_infer_time"] = np.mean(infer_time)
        saved_dict[str(n)][list_names[i]]["std_infer_time"] = np.std(infer_time)
        saved_dict[str(n)][list_names[i]]["global_time"] = end-start
        saved_dict[str(n)][list_names[i]]["nb_steps"] = len(infer_time)

    print("-----> ASM-LU <-----")

    saved_dict[str(n)]["ASM-LU"] = {}

    start = time.time()
    sol, res, infer_time = poisson.solve_cg(    preconditioner = "ASM-LU",
                                                level = level,
                                                stop_mode = "res_rel",
                                                x0 = None,
                                                tol = tolerance,
                                                max_iter = max_iter)
    end = time.time()

    saved_dict[str(n)]["ASM-LU"]["mean_infer_time"] = np.mean(infer_time)
    saved_dict[str(n)]["ASM-LU"]["std_infer_time"] = np.std(infer_time)
    saved_dict[str(n)]["ASM-LU"]["global_time"] = end-start
    saved_dict[str(n)]["ASM-LU"]["nb_steps"] = len(infer_time)

# Serializing json
json_object = json.dumps(saved_dict, indent=2)

# Writing to sample.json
with open(os.path.join(path_savings, "results.json"), "w") as outfile:
    outfile.write(json_object)
outfile.close()