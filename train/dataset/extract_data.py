import numpy as np 
from tqdm import tqdm
import scipy 
from scipy.sparse import csr_matrix, find

import partitioning as part
import resolution as solve

class ExtractDataPDDM():

    def __init__(self, dict_subdata):

        # define mesh_path
        self.mesh_path = dict_subdata["mesh_path"]
        self.path_save_subdomains = dict_subdata["path_save_subdomains"]

        # define number of subdomains and overlap
        self.nb_subdomains = int(np.ceil(dict_subdata["nb_nodes"] / dict_subdata["nb_nodes_subdomain"]))
        self.nb_overlaps = 2

        np.random.seed(dict_subdata["seed"])

        # build dictionnary for partitioning
        self.dict_partition = { "path_mesh_xdmf" : self.mesh_path,
                                "path_save_subdomains" : self.path_save_subdomains,
                                "Nparts" : self.nb_subdomains,
                                "Noverlaps" : self.nb_overlaps
                                }
        
        # initialise the partitioning 
        self.partition = part.Partitioning(self.dict_partition)

        # build the partitioning 
        self.partition.framework_partitioning()

        # part dict 
        self.dict_resolution =  {   "partition" : self.partition,
                                    "path_full_mesh_hdf5": dict_subdata["path_full_mesh_hdf5"],
                                    "path_subdomains": self.path_save_subdomains,
                                    "path_model" : None,
                                    "seed" : dict_subdata["seed"]
                                }
        
        self.poisson = solve.DDMResolution(self.dict_resolution)
      
    def save_list_subdomains_data(self):

        # extract global data
        global_A, global_rhs, global_d2v_map, global_v2d_map = self.poisson.extract_global_problem()
        dict_global = {"global_A" : global_A,
                       "global_rhs" : global_rhs,
                       "global_d2v_map" : global_d2v_map,
                       "global_v2d_map" : global_v2d_map
                       }

        # dictionnary of data
        dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.poisson.extract_fenics_data_subdomains()
        
        dict_stiffness_bc = self.poisson.dict_stiffness_with_bc(dict_A,
                                                                dict_d2v_map, 
                                                                self.partition.dict_restriction,
                                                                self.partition.dict_is_interface)

        dict_preconditioner = { "dict_stiffness_with_bc" : dict_stiffness_bc,
                                "dict_rhs" : dict_rhs,
                                "dict_d2v_map" : dict_d2v_map,
                                "dict_v2d_map" : dict_v2d_map
                                }
        
        # coarse problem data
        Z = self.poisson.compute_nicolaides_operator()
        Z = Z[global_d2v_map]
        Q, Qinv = self.poisson.compute_coarse_operators(global_A, Z)
        dict_preconditioner["nicolaides_operator"] = Z
        dict_preconditioner["inv_coarse_operator"] = Qinv

        # define initial global solution
        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())
        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]
        global_sol[index] = self.poisson.g_array[index]

        list_A_sparse_matrix, list_b_matrix, list_sol, list_prb_data, list_tags, list_coordinates = self.cg_preconditioner(dict_global, dict_preconditioner, global_sol, tol = 1.e-12, max_iter = 500)

        return list_A_sparse_matrix, list_b_matrix, list_sol, list_prb_data, list_tags, list_coordinates
    
    def cg_preconditioner(self, dict_global, dict_preconditioner, x0, tol = 1e-9, max_iter = 1000):

        # list to store the data        
        list_A_sparse_matrix, list_b_matrix, list_sol, list_prb_data, list_tags, list_coordinates = [], [], [], [], [], []

        # store solution
        store_sol = {}
        store_residual = {}

        # extract data
        A = dict_global["global_A"]
        rhs = dict_global["global_rhs"]
        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]

        #initial guess
        x = (x0.copy())[global_d2v_map]

        # initial residual
        r = (rhs - A @ x.reshape(-1,1)).flatten()

        # solve first ASM
        z, dict_sub_sol, dict_rhs, dict_prb_data, dict_tags, dict_dof_coordinates = self.ASM_preconditioner(r.copy()[global_v2d_map], dict_global, dict_preconditioner)
        z = z[global_d2v_map]

        list_A_sparse_matrix.append(dict_preconditioner["dict_stiffness_with_bc"])
        list_b_matrix.append(dict_rhs)
        list_sol.append(dict_sub_sol)
        list_prb_data.append(dict_prb_data)
        list_tags.append(dict_tags)
        list_coordinates.append(dict_dof_coordinates)

        p = z.copy()

        rho_prev = 1

        for iter in tqdm(range(max_iter)):

            rho_prev = np.dot(r, z)

            q = (A @ p.reshape(-1,1)).flatten()

            alpha = rho_prev / np.dot(p, q)
            
            x = x + alpha * p
            r = r - alpha * q

            store_sol[str(iter)] = (x.copy())[global_v2d_map] 

            store_residual[str(iter)] = np.mean(r**2)

            if store_residual[str(iter)] < tol :
                break;

            z, dict_sub_sol, dict_rhs, dict_prb_data, dict_tags, dict_dof_coordinates = self.ASM_preconditioner(r.copy()[global_v2d_map], dict_global, dict_preconditioner)
            z = z[global_d2v_map]
            
            rho = np.dot(r, z)
            
            beta = rho / rho_prev

            p = z + beta*p
            
            rho_prev = rho

            list_A_sparse_matrix.append(dict_preconditioner["dict_stiffness_with_bc"])
            list_b_matrix.append(dict_rhs)
            list_sol.append(dict_sub_sol)
            list_prb_data.append(dict_prb_data)
            list_tags.append(dict_tags)
            list_coordinates.append(dict_dof_coordinates)

        print("PDDM Method has converged in {} iterations with last residual = {}".format(iter, store_residual[str(iter)]))    
        
        return list_A_sparse_matrix, list_b_matrix, list_sol, list_prb_data, list_tags, list_coordinates

    def ASM_preconditioner(self, rhs, dict_global, dict_preconditioner):

        A = dict_global["global_A"]
        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]

        dict_stiffness_with_bc = dict_preconditioner["dict_stiffness_with_bc"]
        dict_d2v_map = dict_preconditioner["dict_d2v_map"]
        dict_v2d_map = dict_preconditioner["dict_v2d_map"]

        Z = dict_preconditioner["nicolaides_operator"]
        Qinv = dict_preconditioner["inv_coarse_operator"]

        coarse_solution = self.poisson.compute_coarse_correction(rhs, Z, Qinv, global_d2v_map, global_v2d_map)

        extended_correction = np.zeros(self.partition.full_graph.number_of_nodes())

        dict_sub_sol, dict_rhs, dict_prb_data, dict_tags, dict_dof_coordinates = {}, {}, {}, {}, {}

        rhs_norm = np.linalg.norm(rhs)

        for part in range(self.partition.Nparts):

            stiffness = dict_stiffness_with_bc[str(part)]
            
            restricted_residual = self.poisson.compute_restricted_residual( part,
                                                                            rhs/rhs_norm,
                                                                            self.partition.full_graph, 
                                                                            dict_d2v_map, 
                                                                            self.partition.dict_restriction, 
                                                                            self.partition.dict_is_interface
                                                                            )
            dict_rhs[str(part)] = restricted_residual
            
            restricted_correction = scipy.sparse.linalg.spsolve(stiffness, restricted_residual)

            dict_sub_sol[str(part)] = restricted_correction

            d2v = dict_d2v_map[str(part)]
            restriction = self.partition.dict_restriction[str(part)]
            interface = self.partition.dict_is_interface[str(part)] 

            prb_data = restricted_residual.reshape(-1,1)
            # prb_data = np.hstack((restricted_residual.reshape(-1,1), np.zeros(np.shape(restricted_residual.reshape(-1,1))), np.zeros(np.shape(restricted_residual.reshape(-1,1)))))
            tags = np.zeros_like(restriction).reshape(-1,1)

            sub_dof_full_boundary = (self.partition.full_graph.domain_boundary_tags[restriction])[d2v]
            sub_dof_interface = (interface[restriction])[d2v]
            sub_dof_coordinates = (self.partition.full_graph.points[restriction][d2v])

            index_tags_boundary = np.where((sub_dof_full_boundary == 101) | (sub_dof_interface == True))[0]
            tags[index_tags_boundary,:] = 1

            dict_prb_data[str(part)] = prb_data
            dict_tags[str(part)] = tags 
            dict_dof_coordinates[str(part)] = sub_dof_coordinates

            ext_corr = self.poisson.extend_partition_of_unity_ASM(  part, 
                                                                    restricted_correction, 
                                                                    self.partition.nodes_membership_overlap, 
                                                                    dict_v2d_map, 
                                                                    self.partition.dict_restriction)
            
            extended_correction = extended_correction + ext_corr

        extended_correction = extended_correction * rhs_norm 
        
        full_solution = extended_correction + coarse_solution
        
        return full_solution, dict_sub_sol, dict_rhs, dict_prb_data, dict_tags, dict_dof_coordinates