import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import matplotlib.pyplot as plt
from copy import deepcopy

import mesh_random as msh 
import json 

import partitioning_original as pt
import resolution_optimized as solve

path_model = "all_results/k30_d10/ckpt/best_model.pt"
mesh_radius = [0.6, 1.0, 2.2]
size_subdomains = 1000
overlap = 2

mean_nodes = []
std_nodes = []

mean_subdomain = []
std_subdomain = []

saved_dict = {}

path_savings = "results/multiple_prbs/"
if not os.path.exists(path_savings):
    os.mkdir(path_savings)

for r in mesh_radius :
    
    saved_dict[str(r)] = {}
    
    nnodes, nsub = [], []
    iter_gnn, iter_lu, iter_cg = [], [], []

    for k in range(100):

        print("Radius {}, Iteration {}".format(r, k))

        mesh_dict = {   "folder_to_save" : os.path.join(path_savings, "folder_mesh"),
                        "name" : "mesh",
                        "radius" : r,
                        "nb_boundary_points" : 20,
                        "hsize" : 0.02,
                        "dirichlet_tag" : 101,
                        "seed" : k
                    }
        
        myMesh = msh.MyMesh(mesh_dict)
        myMesh.generate()

        part_dict = {   "path_mesh_xdmf" : os.path.join(path_savings, "folder_mesh", "mesh.xdmf"),
                        "path_save_subdomains" : os.path.join(path_savings, "subdomains"),
                        "Nparts" : int(np.ceil(myMesh.nb_nodes /  size_subdomains)),
                        "Noverlaps" : overlap
                    }

        nnodes.append(myMesh.nb_nodes)

        partition = pt.Partitioning(part_dict)
        partition.framework_partitioning()

        Nsub = int(np.mean([len(partition.dict_restriction[str(i)]) for i in range(part_dict["Nparts"])]))
        print('Number of nodes per subdomains : ', Nsub)
        
        nsub.append(part_dict["Nparts"])

        dict_resolution = { "partition" : partition,
                            "path_full_mesh_hdf5" : os.path.join(path_savings, "folder_mesh", "mesh_fenics.h5"),
                            "path_subdomains" : os.path.join(path_savings, "subdomains"),
                            "path_model" : path_model,
                            "radius" : r,
                            "seed" : k,
                            "path_savings": path_savings
                            }
        
        poisson = solve.DDMResolution(dict_resolution)

        print("Step 5 ---> Solve with PCG-DDML-DSS")
        sol_gnn, res_gnn, infer_time_gnn = poisson.solve_cg(    preconditioner = "ASM-GNN",
                                                                level = 2,
                                                                stop_mode = "res_rel",
                                                                x0 = None,
                                                                tol = 1.e-6,
                                                                max_iter=500)

        print("Step 6 ---> Solve with PCG-ASM-LU")
        sol_lu, res_lu, infer_time_lu = poisson.solve_cg(   preconditioner = "ASM-LU",
                                                            level = 2,
                                                            stop_mode = "res_rel",
                                                            x0 = None,
                                                            tol = 1.e-6,
                                                            max_iter=500)

        print("Step 6 ---> Solve with CG")
        sol_cg, res_cg, infer_time_cg = poisson.solve_cg(   preconditioner = None,
                                                            level = 2,
                                                            stop_mode = "res_rel",
                                                            x0 = None,
                                                            tol = 1.e-6,
                                                            max_iter=500)

        iter_gnn.append(len(res_gnn))
        iter_lu.append(len(res_lu))
        iter_cg.append(len(res_cg))
    
    saved_dict[str(r)]["mean_nodes"] = np.mean(nnodes)
    saved_dict[str(r)]["std_nodes"] = np.std(nnodes)
    saved_dict[str(r)]["mean_sub"] = np.mean(nsub)
    saved_dict[str(r)]["std_sub"] = np.std(nsub)
    saved_dict[str(r)]["mean_iter_gnn"] = np.mean(iter_gnn)
    saved_dict[str(r)]["std_iter_gnn"] = np.std(iter_gnn)
    saved_dict[str(r)]["mean_iter_lu"] = np.mean(iter_lu)
    saved_dict[str(r)]["std_iter_lu"] = np.std(iter_lu)
    saved_dict[str(r)]["mean_iter_cg"] = np.mean(iter_cg)
    saved_dict[str(r)]["std_iter_cg"] = np.std(iter_cg)

    # Serializing json
    json_object = json.dumps(saved_dict, indent=3)
    
    # Writing to sample.json
    with open(os.path.join(path_savings, "results.json"), "w") as outfile:
        outfile.write(json_object)
    outfile.close()
