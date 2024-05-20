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
# import mesh_f1 as msh 
import partitioning_original as pt
import resolution_optimized as solve
import model_dss_inference as modtest

###############################
###############################
###############################

### C++ COMPATIBILITY ###
seed = 1234
radius = 1.2
size_subdomains = 1000
overlap = 1
path_savings = "results/test_k5_d5/"
model_name = "k5_d5"
path_model = os.path.join("all_results", model_name, "ckpt", "new_model.pt")
tolerance = 1.e-3
max_iter = 50
level = 2

### MESH F1 ###
# seed = 1234
# radius = 9.0
# size_subdomains = 1000
# overlap = 2
# path_savings = "results/test_f1/"
# model_name = "k30_d10"
# path_model = os.path.join("all_results", model_name, "ckpt", "new_model.pt")
# tolerance = 1.e-6
# max_iter = 1000
# level = 2

###############################
###############################
###############################

def custom_sort(item):
    # Extract numbers after 'k' and 'd'
    k_number = int(re.search(r'k(\d+)', item).group(1))
    d_number = int(re.search(r'd(\d+)', item).group(1))
    return k_number, d_number

if not os.path.exists(path_savings):
    os.mkdir(path_savings)  

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

    with open(os.path.join(path_savings, 'partition.pkl'), 'wb') as file:
        pickle.dump(partition, file)
    print(f'Object successfully saved to "partition.pkl"')

else :

    with open(os.path.join(path_savings, 'partition.pkl'), "rb") as file:
        partition = pickle.load(file)

print("-----> GPU WARMUP <-----")
dict_resolution = { "partition" : partition,
                    "path_full_mesh_hdf5" : os.path.join(path_savings, "mesh", "temp_fenics.h5"),
                    "path_subdomains" : os.path.join(path_savings, "subdomains"),
                    "path_model" : path_model,
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
                                            max_iter=4
                                            )
print("Warmup time : ", infer_time)

print("-----> START <-----")

saved_dict = {}

saved_dict[model_name] = {}

print(f"Solve for model {model_name}")

dict_resolution = { "partition" : partition,
                    "path_full_mesh_hdf5" : os.path.join(path_savings, "mesh", "temp_fenics.h5"),
                    "path_subdomains" : os.path.join(path_savings, "subdomains"),
                    "path_model" : path_model,
                    "radius" : radius,
                    "seed" : seed,
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

saved_dict[model_name]["mean_infer_time"] = np.mean(infer_time)
saved_dict[model_name]["std_infer_time"] = np.std(infer_time)
saved_dict[model_name]["global_time"] = end-start
saved_dict[model_name]["residual"] = res
saved_dict[model_name]["nb_steps"] = len(infer_time)

res1 = [res[str(i)] for i in range(len(res))]

print("-----> ASM-LU <-----")

saved_dict["ASM-LU"] = {}

start = time.time()
sol, res, infer_time = poisson.solve_cg(    preconditioner = "ASM-LU",
                                            level = level,
                                            stop_mode = "res_rel",
                                            x0 = None,
                                            tol = tolerance,
                                            max_iter = max_iter)
end = time.time()

saved_dict["ASM-LU"]["mean_infer_time"] = np.mean(infer_time)
saved_dict["ASM-LU"]["std_infer_time"] = np.std(infer_time)
saved_dict["ASM-LU"]["global_time"] = end-start
saved_dict["ASM-LU"]["residual"] = res
saved_dict["ASM-LU"]["nb_steps"] = len(infer_time)

res2 = [res[str(i)] for i in range(len(res))]

plt.plot(res2, label="ASM-LU")
plt.plot(res1, label="ASM-GNN")
plt.yscale("log")
plt.show()
