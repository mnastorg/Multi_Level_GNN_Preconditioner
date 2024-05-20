### PACKAGES ###
import sys 
sys.path.append("..")

import os
import numpy as np
import scipy
from fenics import *
from scipy.sparse import find, csr_matrix, SparseEfficiencyWarning
from copy import deepcopy
from tqdm import tqdm 
import warnings

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

warnings.simplefilter('ignore',SparseEfficiencyWarning)
################

class DDMResolution():

    def __init__(self, dict_resolution):

        self.partition = dict_resolution["partition"]
        self.path_full_mesh = dict_resolution["path_full_mesh_hdf5"]
        self.path_subdomains = dict_resolution["path_subdomains"]

        self.path_ml_model = dict_resolution["path_model"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        np.random.seed(dict_resolution["seed"])
        
        # force function and boundary value
        self.param_f = np.random.uniform(-10, 10, 3)
        self.param_g = np.random.uniform(-10, 10, 6)

        # define expressions of functions f and g
        L = 1
        self.f_expression = Expression( 'A*((x[0]/L)-1)*((x[0]/L)-1) + B*(x[1]/L)*(x[1]/L) + C',
                                        A = self.param_f[0], B = self.param_f[1], C = self.param_f[2], L = L,
                                        degree = 2 
                                        )
        self.g_expression = Expression( 'A*(x[0]/L)*(x[0]/L) + B*(x[0]/L)*(x[1]/L) + C*(x[1]/L)*(x[1]/L) + D*(x[0]/L) + E*(x[1]/L) + F',
                                        A = self.param_g[0], B = self.param_g[1], C = self.param_g[2], D = self.param_g[3], E = self.param_g[4], F = self.param_g[5], L = L,
                                        degree = 2 
                                        )

        # read full mesh
        comm = MPI.comm_world
        self.full_mesh = Mesh()
        with HDF5File(comm, self.path_full_mesh, "r") as h5file:
                h5file.read(self.full_mesh, "mesh", False)
                self.full_facet = MeshFunction("size_t", self.full_mesh, self.full_mesh.topology().dim() - 1)
                h5file.read(self.full_facet, "facet")
        
        # array of f and g functions in vertex coordinates
        self.f_array = self.f_expression.compute_vertex_values(self.full_mesh) 
        # print("Force function", self.f_array)   
        self.g_array = self.g_expression.compute_vertex_values(self.full_mesh)    
        # print("Boundary function", self.g_array)   

    def solve_global_problem(self):
        '''
        Returns the solution (sol) of a Poisson problem 
            - Delta(u)  = f in Omega
                    u   = g in boundaryOmega 
        computed with Fenics on the full mesh 
        '''

        V = FunctionSpace(self.full_mesh, "Lagrange", 1)

        u = TrialFunction(V)
        v = TestFunction(V)

        bc = DirichletBC(V, self.g_expression, self.full_facet, 101)

        a = inner(grad(u), grad(v))*dx
        L = self.f_expression*v*dx

        A = assemble(a)
        rhs = assemble(L)

        bc.apply(A, rhs)

        u = Function(V)
        solve(A, u.vector(), rhs)

        sol = u.compute_vertex_values(self.full_mesh)

        return sol
    
    #########################################
    ###### HIGH LEVEL SCHWARZ METHODS  ######
    #########################################

    def solve_jacobi_schwarz(self, level = 1, stop_mode = "norm", tol = 1.e-6, max_iter = 500):
        """
        Full pipeline to solve the Jacobi-Schwarz problem.
        - Use level = 1 for classical jacobi-schwarz 
        - Use level = 2 for 2-level method (i.e Nicolaides coarse space correction).
        - stop_mode = "norm" to stop at the norm of the residual
        - stop_mode = "mse" to stop at the MSE of the residual.
        Outputs dictionnary of solution and residual at each iteration.
        until convergence.
        """

        if (stop_mode != 'norm') & (stop_mode != 'mse') :
            sys.exit("stop_mode should be norm or mse.")

        if (level != 1) & (level != 2) :
            sys.exit("level should be 1 or 2.")

        # extract submatrices
        dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()

        dict_stiffness_with_bc = self.dict_stiffness_with_bc(dict_A,
                                                             dict_d2v_map,
                                                             self.partition.dict_restriction,
                                                             self.partition.dict_is_interface)
        
        # extract global matrices
        global_A, global_rhs, global_d2v_map, global_v2d_map = self.extract_global_problem()

        if level == 2 :
            # compute nicolaides operator 
            Z = self.compute_nicolaides_operator()
            Z = Z[global_d2v_map]
            # compute coarse matrix and its inverse 
            Q, Qinv = self.compute_coarse_operators(global_A, Z)

        # save dictionnary
        store_sol = {}
        store_residual = {}

        # initialize global solution
        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]
        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())
        global_sol[index] = self.g_array[index]
                                   
        for iter in tqdm(range(max_iter)):

            residual_vector = global_rhs - global_A @ global_sol[global_d2v_map].reshape(-1,1)          
            residual_vector = residual_vector[global_v2d_map].flatten()          
            
            if stop_mode == "norm" : 
                residual = np.linalg.norm(residual_vector)
            elif stop_mode == "mse" :
                residual = np.mean(residual_vector**2)
            
            store_sol[str(iter)] = global_sol
            store_residual[str(iter)] = residual

            if store_residual[str(iter)] < tol : 
                break;
            
            if level == 2 :

                coarse_solution = self.compute_coarse_correction(residual_vector, Z, Qinv, global_d2v_map, global_v2d_map)
                coarse_solution[index] = 0

                global_sol += coarse_solution

            # update rhs vector using the global solution
            dict_rhs = self.update_rhs( self.partition.full_graph, 
                                        dict_rhs, 
                                        global_sol, 
                                        dict_d2v_map, 
                                        dict_v2d_map, 
                                        self.partition.dict_restriction, 
                                        self.partition.dict_is_interface)
            
            new_solution = np.zeros(self.partition.full_graph.number_of_nodes())            
            
            # solve for each subdomain
            for part in range(self.partition.Nparts):
                
                stiffness = dict_stiffness_with_bc[str(part)]
                rhs = dict_rhs[str(part)]
                sub_sol = scipy.sparse.linalg.spsolve(stiffness, rhs)

                ext_sol = self.extend_partition_of_unity_RAS(part, 
                                                             sub_sol, 
                                                             self.partition.nodes_membership_overlap, 
                                                             dict_v2d_map, 
                                                             self.partition.dict_restriction)

                new_solution += ext_sol

            global_sol = new_solution

        return store_sol, store_residual

    def solve_iterative_RAS(self, level = 1, stop_mode = "norm", tol = 1.e-6, max_iter = 500):
        """
        Full pipeline to solve the iterative RAS problem.
        - Use level = 1 for classical iterative RAS algorithm
        - Use level = 2 for 2-level method (i.e Nicolaides coarse space correction).
        - stop_mode = "norm" to stop at the norm of the residual
        - stop_mode = "mse" to stop at the MSE of the residual.
        Outputs dictionnary of solution and residual at each iteration.
        until convergence.        
        """

        if (stop_mode != 'norm') & (stop_mode != 'mse') :
            sys.exit("stop_mode should be norm or mse")

        if (level != 1) & (level != 2) :
            sys.exit("level should be 1 or 2.")

        dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()
        
        dict_stiffness_with_bc = self.dict_stiffness_with_bc(dict_A,
                                                             dict_d2v_map,
                                                             self.partition.dict_restriction,
                                                             self.partition.dict_is_interface)

        global_A, global_rhs, global_d2v_map, global_v2d_map = self.extract_global_problem()

        if level == 2 :
            # compute nicolaides operator 
            Z = self.compute_nicolaides_operator()
            Z = Z[global_d2v_map]
            # compute coarse matrix and its inverse 
            Q, Qinv = self.compute_coarse_operators(global_A, Z)

        # save dictionnary
        store_sol = {}
        store_residual = {}

        # initialize global solution
        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]
        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())
        global_sol[index] = self.g_array[index]

        for iter in tqdm(range(max_iter)):

            residual_vector = global_rhs - global_A @ global_sol[global_d2v_map].reshape(-1,1)          
            residual_vector = residual_vector[global_v2d_map].flatten()          
            
            if stop_mode == "norm" : 
                residual = np.linalg.norm(residual_vector)
            elif stop_mode == "mse" :
                residual = np.mean(residual_vector**2)
            
            store_sol[str(iter)] = global_sol
            store_residual[str(iter)] = residual

            if store_residual[str(iter)] < tol : 
                break;

            if level == 2 :

                coarse_solution = self.compute_coarse_correction(residual_vector, Z, Qinv, global_d2v_map, global_v2d_map)
                # coarse_solution[index] = 0
                
                global_sol = global_sol + coarse_solution

                residual_vector = global_rhs - global_A @ global_sol[global_d2v_map].reshape(-1,1)          
                residual_vector = residual_vector[global_v2d_map].flatten()          

            extended_correction = np.zeros(self.partition.full_graph.number_of_nodes())
            
            for part in range(self.partition.Nparts):

                stiffness = dict_stiffness_with_bc[str(part)]
                
                restricted_residual = self.compute_restricted_residual( part,
                                                                        residual_vector,
                                                                        self.partition.full_graph, 
                                                                        dict_d2v_map, 
                                                                        self.partition.dict_restriction, 
                                                                        self.partition.dict_is_interface
                                                                        )
                
                restricted_correction = scipy.sparse.linalg.spsolve(stiffness, restricted_residual)
                
                ext_corr = self.extend_partition_of_unity_RAS(  part, 
                                                                restricted_correction, 
                                                                self.partition.nodes_membership_overlap, 
                                                                dict_v2d_map, 
                                                                self.partition.dict_restriction)
                
                extended_correction = extended_correction + ext_corr

            global_sol = global_sol + extended_correction

            # if level == 1 :
            #     global_sol = global_sol + extended_correction

            # elif level == 2 :
            #     global_sol = global_sol + extended_correction + coarse_solution

        return store_sol, store_residual
    
    def solve_krylov(self, solver = "cg", preconditioner = None, level = 1, stop_mode = "norm", tol = 1.e-6, max_iter = 500):
        """
        Solve the linear system arising from Poisson problem discretization
        using a Krylov method.
        - solver = "cg" for conjugate gradient or 
                 = "bicgstab" for biconjugate gradient stabilized.
        - preconditioner = None for no preconditioner
                         = "ASM" for ASM preconditioner (only with CG)
                         = "RAS" for RAS preconditioner (only with BICGSTAB)
        if preconditioner is not None, level = 1 or level = 2 for resp. the 
        classical 1-level method or coarse space correction (2-level).
        - stop_mode = "norm" to stop at the norm of the residual
                    = "mse" to stop at the MSE of the residual.
        Outputs dictionnary of solution and residual at each iteration.
        until convergence.        
        """

        ### condition to run the method ###

        if (solver != 'cg') & (solver != 'bicgstab') :
            sys.exit("solver should be cg or bicgstab")

        if (preconditioner != None) & (preconditioner != "ASM") & (preconditioner != "RAS"):
            sys.exit("preconditioner should be None or ASM or RAS")

        if (solver == "bicgstab") & (preconditioner == "ASM"):
            sys.exit("can't use bicgstab with ASM preconditioner")

        if (solver == "cg") & (preconditioner == "RAS"):
            sys.exit("can't use cg with RAS preconditioner")
            
        if (stop_mode != 'norm') & (stop_mode != 'mse') :
            sys.exit("stop_mode should be norm or mse")

        if (level != 1) & (level != 2) :
            sys.exit("level should be 1 or 2.")

        ####################################

        # extract global data
        global_A, global_rhs, global_d2v_map, global_v2d_map = self.extract_global_problem()

        dict_global = {"global_A" : global_A,
                       "global_rhs" : global_rhs,
                       "global_d2v_map" : global_d2v_map,
                       "global_v2d_map" : global_v2d_map
                       }

        # extract preconditioner data is needed
        if preconditioner is not None : 

            dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()
            
            dict_stiffness_with_bc = self.dict_stiffness_with_bc(dict_A,
                                                                dict_d2v_map,
                                                                self.partition.dict_restriction,
                                                                self.partition.dict_is_interface)
            dict_preconditioner = { "dict_stiffness_with_bc" : dict_stiffness_with_bc,
                                    "dict_rhs" : dict_rhs,
                                    "dict_d2v_map" : dict_d2v_map,
                                    "dict_v2d_map" : dict_v2d_map
                                    }
            # if preconditioner and level == 2 extract coarse space data
            if level == 2 : 
                # compute nicolaides operator 
                Z = self.compute_nicolaides_operator()
                Z = Z[global_d2v_map]
                # compute coarse matrix and its inverse 
                Q, Qinv = self.compute_coarse_operators(global_A, Z)

                dict_preconditioner["nicolaides_operator"] = Z
                dict_preconditioner["inv_coarse_operator"] = Qinv

        # define initial global solution
        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())
        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]
        global_sol[index] = self.g_array[index]

        if preconditioner == None :
            if solver == "cg" : 
                solution, residual = self.cg(dict_global, global_sol, stop_mode = stop_mode, tol = tol, max_iter = max_iter)
            elif solver == "bicgstab":
                solution, residual = self.bicgstab(dict_global, global_sol, stop_mode = stop_mode, tol = tol, max_iter = max_iter)

        if (solver == "cg") & (preconditioner == "ASM"):
            solution, residual = self.cg_preconditioner(dict_global, dict_preconditioner, global_sol, level = level, stop_mode = stop_mode, tol = tol, max_iter = max_iter)

        if (solver == "bicgstab") & (preconditioner == "RAS"):
            solution, residual = self.bicgstab_preconditioner(dict_global, dict_preconditioner, global_sol, level = level, stop_mode = stop_mode, tol = tol, max_iter = max_iter)
                
        return solution, residual

    #########################################
    ###### HIGH LEVEL ML DDM SOLVER  ########
    #########################################

    def solve_one_level_ddm_with_ml(self, tol = 1.e-6, max_iter = 500):

        dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()
        
        dict_stiffness_with_bc = self.dict_stiffness_with_bc(dict_A,
                                                             dict_d2v_map,
                                                             self.partition.dict_restriction,
                                                             self.partition.dict_is_interface)
        store_sol = {}
        
        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())

        for iter in tqdm(range(max_iter)):
            
            dict_rhs = self.update_rhs( self.partition.full_graph, 
                                        dict_rhs, 
                                        global_sol, 
                                        dict_d2v_map, 
                                        dict_v2d_map, 
                                        self.partition.dict_restriction, 
                                        self.partition.dict_is_interface
                                         )

            batch_config = {"dict_stiffness" : dict_stiffness_with_bc,
                            "dict_rhs" : dict_rhs,
                            "dict_d2v_map" : dict_d2v_map,
                            "dict_v2d_map" : dict_v2d_map,
                            "global_sol" : global_sol}
            
            data_list = self.construct_batch(batch_config)

            loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
            
            batch_solution = self.compute_ml_solution(loader)

            dict_sub_sol_v2d = self.batch_to_individual(data_list, batch_solution, dict_v2d_map)
            
            global_sol = self.compute_global_sol( dict_sub_sol_v2d,
                                                            self.partition.nodes_membership_overlap,
                                                            self.partition.dict_restriction
                                                            )
            
            store_sol[str(iter)] = global_sol

            if iter > 0 :            
                if np.linalg.norm(store_sol[str(iter)] - store_sol[str(iter-1)]) < tol : 
                    break;

        return store_sol        

    def solve_two_level_ddm_with_ml(self, tol = 1.e-6, max_iter = 500):

        dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()
        
        dict_stiffness_with_bc = self.dict_stiffness_with_bc(dict_A,
                                                             dict_d2v_map,
                                                             self.partition.dict_restriction,
                                                             self.partition.dict_is_interface)

        global_A, global_rhs, global_d2v_map, global_v2d_map = self.extract_global_problem()

        # compute nicolaides operator 
        Z = self.compute_nicolaides_operator()
        Z = Z[global_d2v_map]
        # compute coarse matrix and its inverse 
        Q, Qinv = self.compute_coarse_operators(global_A, Z)

        # save dictionnary
        store_sol = {}
        store_residual = {}

        # initialize global solution
        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]
        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())
        global_sol[index] = self.g_array[index]

        for iter in tqdm(range(max_iter)):
            
            residual_vector = global_rhs - global_A @ global_sol[global_d2v_map].reshape(-1,1)          
            residual_vector = residual_vector[global_v2d_map].flatten()          
            
            residual = np.linalg.norm(residual_vector)
            
            store_sol[str(iter)] = global_sol
            store_residual[str(iter)] = residual

            if store_residual[str(iter)] < tol : 
                break;

            coarse_solution = self.compute_coarse_correction(residual_vector, Z, Qinv, global_d2v_map, global_v2d_map)
            coarse_solution[index] = 0
            
            global_sol += coarse_solution

            residual_vector = global_rhs - global_A @ global_sol[global_d2v_map].reshape(-1,1)          
            residual_vector = residual_vector[global_v2d_map].flatten()          

            batch_config = {"dict_stiffness": dict_stiffness_with_bc,
                            "residual"      : residual_vector,
                            "dict_d2v_map"  : dict_d2v_map,
                            "dict_v2d_map"  : dict_v2d_map,
                            }
                        
            data_list = self.construct_batch(batch_config)
            
            loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
            
            batch_solution = self.compute_ml_solution(loader)

            # dict_sub_sol_v2d = self.batch_to_individual(data_list, batch_solution, dict_v2d_map)
            extended_correction = self.batch_to_extend_RAS(data_list, batch_solution, dict_v2d_map)

            global_sol = global_sol + extended_correction

            # global_sol = self.compute_global_sol( dict_sub_sol_v2d,
            #                                                 self.partition.nodes_membership_overlap,
            #                                                 self.partition.dict_restriction
            #                                                 )

            # store_sol[str(iter)] = global_sol

            # if iter > 0 :            
            #     if np.linalg.norm(store_sol[str(iter)] - store_sol[str(iter-1)]) < tol : 
            #         break;

        return store_sol        
       
    def solve_mixed_ddm_ml(self, tol = 1.e-6, max_iter = 500):

        dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()
        
        dict_stiffness_with_bc = self.dict_stiffness_with_bc(dict_A,
                                                             dict_d2v_map,
                                                             self.partition.dict_restriction,
                                                             self.partition.dict_is_interface)

        store_sol = {}
        dict_sub_sol = {}

        global_sol = np.zeros(self.partition.full_graph.number_of_nodes())

        mode = "ML"
        cumul_err = [1.e4]

        for iter in tqdm(range(max_iter)):
            
            dict_rhs = self.update_rhs( self.partition.full_graph, 
                                        dict_rhs, 
                                        global_sol, 
                                        dict_d2v_map, 
                                        dict_v2d_map, 
                                        self.partition.dict_restriction, 
                                        self.partition.dict_is_interface
                                        )

            if iter == 0 : 

                batch_config = {"dict_stiffness" : dict_stiffness_with_bc,
                                "dict_rhs" : dict_rhs,
                                "dict_d2v_map" : dict_d2v_map,
                                "dict_v2d_map" : dict_v2d_map,
                                "global_sol" : global_sol}
                
                data_list = self.construct_batch(batch_config)
                loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
                
                batch_solution = self.compute_ml_solution(loader)

                dict_sub_sol_v2d = self.batch_to_individual(data_list, batch_solution, dict_v2d_map)
                
                global_sol = self.compute_global_sol( dict_sub_sol_v2d,
                                                                self.partition.nodes_membership_overlap,
                                                                self.partition.dict_restriction
                                                                )

                store_sol[str(iter)] = global_sol
            
            else :

                if mode == 'ML' :
            
                    batch_config = {"dict_stiffness" : dict_stiffness_with_bc,
                                    "dict_rhs" : dict_rhs,
                                    "dict_d2v_map" : dict_d2v_map,
                                    "dict_v2d_map" : dict_v2d_map,
                                    "global_sol" : global_sol}
                    
                    data_list = self.construct_batch(batch_config)
                    loader = DataLoader(data_list, batch_size=len(data_list), shuffle=False)
                    
                    batch_solution = self.compute_ml_solution(loader)

                    dict_sub_sol_v2d = self.batch_to_individual(data_list, batch_solution, dict_v2d_map)
                    
                    global_sol = self.compute_global_sol( dict_sub_sol_v2d,
                                                                    self.partition.nodes_membership_overlap,
                                                                    self.partition.dict_restriction
                                                                    )

                    store_sol[str(iter)] = global_sol
                    
                    inc_err = np.linalg.norm(store_sol[str(iter)] - store_sol[str(iter-1)])
                    cumul_err.append(inc_err)
                    
                    if cumul_err[iter - 1] - cumul_err[iter] < 0 :
                        saved_iter = iter
                        mode = 'DDM'

                elif mode == 'DDM':

                    for part in range(self.partition.Nparts):
                        stiffness = dict_stiffness_with_bc[str(part)]
                        rhs = dict_rhs[str(part)]
                        sub_sol = scipy.sparse.linalg.spsolve(stiffness, rhs)
                        dict_sub_sol[str(part)] = sub_sol[dict_v2d_map[str(part)]]

                    global_sol = self.compute_global_sol(  dict_sub_sol,
                                                                self.partition.nodes_membership_overlap,
                                                                self.partition.dict_restriction
                                                                )
                    store_sol[str(iter)] = global_sol

                elif np.linalg.norm(store_sol[str(iter)] - store_sol[str(iter-1)]) < tol:
                    break;

        return store_sol, saved_iter
    
    #########################################
    # LOW LEVEL METHODS FOR GENERAL SCHWARZ #
    #########################################

    def extract_fenics_data_subdomains(self):
        '''
        Returns 4 dictionnaries. For each keys (i.e. subdomain),
        gives the stiffness matrix A, the RHS array and dof to vertex
        mappings (and vice versa).
        '''

        dict_A, dict_rhs = {}, {}
        dict_d2v_map, dict_v2d_map = {}, {}

        for part in range(self.partition.Nparts):
            inter_mesh = Mesh()
            with XDMFFile(os.path.join(self.path_subdomains, "subdomain_{}.xdmf".format(part))) as xdmf:
                xdmf.read(inter_mesh)
            # info(inter_mesh)

            Vsub = FunctionSpace(inter_mesh, "Lagrange", 1)

            u = TrialFunction(Vsub)
            v = TestFunction(Vsub)
            
            a = inner(grad(u), grad(v))*dx
            L = self.f_expression*v*dx

            A = assemble(a)
            B = assemble(L)

            A_sparse_matrix = csr_matrix(A.copy().array())
            B_matrix = B.get_local().reshape(-1,1)

            dict_A[str(part)] = A_sparse_matrix
            dict_rhs[str(part)] = B_matrix
            dict_v2d_map[str(part)] = vertex_to_dof_map(Vsub)
            dict_d2v_map[str(part)] = dof_to_vertex_map(Vsub)

        return dict_A, dict_rhs, dict_v2d_map, dict_d2v_map

    def dict_stiffness_with_bc(self, dict_A_matrix, dict_d2v_map, dict_restriction, dict_is_interface):
        '''
        Returns a dictionnary of stiffness matrix on which Dirichlet
        boundary conditions are applied. Whenever a node is a boundary
        node of a subdomain, it sets 0 on the line except a 1 on the diagonal. 
        '''

        dict = {}
        for part in range(self.partition.Nparts):

            A = deepcopy(dict_A_matrix[str(part)])
            d2v = dict_d2v_map[str(part)]
            ext = dict_restriction[str(part)]
            sub_is_interface = ((dict_is_interface[str(part)])[ext])[d2v]
            sub_domain_boundary_tags = (self.partition.full_graph.domain_boundary_tags[ext])[d2v]
            
            for node in range(np.shape(A)[0]):
                if sub_domain_boundary_tags[node] == 101 or sub_is_interface[node] == True : 
                    A[node,:] = 0
                    A[node, node] = 1
            dict[str(part)] = A

        return dict

    def update_rhs(self, full_graph, dict_rhs_array, global_u, dict_d2v_map, dict_v2d_map, dict_restriction, dict_interface):
        '''
        Returns a dictionnary of updated RHS using the global solution.
        For each keys (i.e. subdomain), returns the RHS array with exact solution
        on the boundary of the global domain and transition solution at interfaces
        between subdomains.
        '''

        dict = {}
        for part in range(self.partition.Nparts):
            rhs = deepcopy(dict_rhs_array[str(part)])
            d2v = dict_d2v_map[str(part)]
            v2d = dict_v2d_map[str(part)]
            restriction = dict_restriction[str(part)]
            interface = dict_interface[str(part)]

            rhs_mesh_order = rhs[v2d]
            for node in range(len(rhs_mesh_order)):
                if full_graph.domain_boundary_tags[restriction[node]] == 101 :
                    rhs_mesh_order[node] = self.g_array[restriction[node]]
                elif full_graph.domain_boundary_tags[restriction[node]] != 101 and interface[restriction[node]] == True :
                    rhs_mesh_order[node] = global_u[restriction[node]]
            
            rhs_final = rhs_mesh_order[d2v]

            dict[str(part)] = rhs_final

        return dict

    def compute_restricted_residual(self, part, residual, full_graph, dict_d2v_map, dict_restriction, dict_interface):
        '''
        Returns the restricted residual to a subdomain "part".
        Also add 0 whenever a node is on the boundary.
        '''

        d2v = dict_d2v_map[str(part)]
        restriction = dict_restriction[str(part)]
        interface = dict_interface[str(part)]

        # residual restricted to part subdomain in vertex order
        new_residual = residual[restriction]
        # convert restricted residual to subdomain order
        new_residual = new_residual[d2v]
        # put 0 if a node is at the boundary of the domain
        # first convert interface and boundary tag to subdomain order
        boundary_tag = (full_graph.domain_boundary_tags[restriction])[d2v]
        is_interface = (interface[restriction])[d2v]
        # select index 
        index_boundary = np.where((boundary_tag == 101) | (is_interface == True))[0]
        # put 0 in the residual vector
        new_residual[index_boundary] = 0

        return new_residual
     
    def extend_partition_of_unity_RAS(self, part, subvector, nodes_membership_overlap, dict_v2d_map, dict_restriction):
        '''
        Returns the extension of a subvector of part subdomain
        computed with the extension of unity.
        '''

        # empty array to store extended solution
        extended_array = np.zeros(self.partition.full_graph.number_of_nodes())

        # additional information
        v2d = dict_v2d_map[str(part)]
        restriction = dict_restriction[str(part)]
        # convert subvector to vertex order
        ordered_vector = subvector[v2d]
        
        for node in range(len(ordered_vector)):
            partition = (nodes_membership_overlap[restriction[node]])
            card = len(partition)
            extended_array[restriction[node]] = ordered_vector[node]/card

        return extended_array

    def extend_partition_of_unity_ASM(self, part, subvector, nodes_membership_overlap, dict_v2d_map, dict_restriction):
        '''
        Returns the extension of a subvector of part subdomain
        computed with no extension of unity.
        '''
        # empty array to store extended solution
        extended_array = np.zeros(self.partition.full_graph.number_of_nodes())

        # additional information
        v2d = dict_v2d_map[str(part)]
        restriction = dict_restriction[str(part)]
        # convert restricted_correction to vertex order
        ordered_correction = subvector[v2d]
        
        for node in range(len(ordered_correction)):
            extended_array[restriction[node]] = ordered_correction[node]

        return extended_array

    def ASM_preconditioner(self, rhs, dict_global, dict_preconditioner, level = 1):
        '''
        Compute the 1 or 2 level ASM Preconditioner applied to
        rhs vector.
        Outputs dictionnary of solution at each iteration.
        '''

        A = dict_global["global_A"]
        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]

        dict_stiffness_with_bc = dict_preconditioner["dict_stiffness_with_bc"]
        dict_d2v_map = dict_preconditioner["dict_d2v_map"]
        dict_v2d_map = dict_preconditioner["dict_v2d_map"]

        # index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]

        if level == 2 :
            # compute nicolaides operator 
            Z = dict_preconditioner["nicolaides_operator"]
            Qinv = dict_preconditioner["inv_coarse_operator"]

            coarse_solution = self.compute_coarse_correction(rhs, Z, Qinv, global_d2v_map, global_v2d_map)
            # coarse_solution[index] = 0

        extended_correction = np.zeros(self.partition.full_graph.number_of_nodes())

        for part in range(self.partition.Nparts):

            ## solve local subproblems
            stiffness = dict_stiffness_with_bc[str(part)]
            
            restricted_residual = self.compute_restricted_residual( part,
                                                                    rhs,
                                                                    self.partition.full_graph, 
                                                                    dict_d2v_map, 
                                                                    self.partition.dict_restriction, 
                                                                    self.partition.dict_is_interface
                                                                    )
            
            restricted_correction = scipy.sparse.linalg.spsolve(stiffness, restricted_residual)
            
            ext_corr = self.extend_partition_of_unity_ASM(  part, 
                                                            restricted_correction, 
                                                            self.partition.nodes_membership_overlap, 
                                                            dict_v2d_map, 
                                                            self.partition.dict_restriction)
            
            extended_correction = extended_correction + ext_corr

        if level == 1 :         
            full_solution = extended_correction
        elif level == 2 : 
            full_solution = extended_correction + coarse_solution
        
        return full_solution

    def RAS_preconditioner(self, rhs, dict_global, dict_preconditioner, level = 1):
        '''
        Compute the 1 or 2 level ASM Preconditioner applied to
        rhs vector.
        Outputs dictionnary of solution at each iteration.
        '''

        A = dict_global["global_A"]
        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]

        dict_stiffness_with_bc = dict_preconditioner["dict_stiffness_with_bc"]
        dict_d2v_map = dict_preconditioner["dict_d2v_map"]
        dict_v2d_map = dict_preconditioner["dict_v2d_map"]

        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]

        if level == 2 :
            # compute nicolaides operator 
            Z = dict_preconditioner["nicolaides_operator"]
            Qinv = dict_preconditioner["inv_coarse_operator"]

            coarse_solution = self.compute_coarse_correction(rhs, Z, Qinv, global_d2v_map, global_v2d_map)
            # coarse_solution[index] = 0

        extended_correction = np.zeros(self.partition.full_graph.number_of_nodes())

        for part in range(self.partition.Nparts):

            ## solve local subproblems
            stiffness = dict_stiffness_with_bc[str(part)]
            
            restricted_residual = self.compute_restricted_residual( part,
                                                                    rhs,
                                                                    self.partition.full_graph, 
                                                                    dict_d2v_map, 
                                                                    self.partition.dict_restriction, 
                                                                    self.partition.dict_is_interface
                                                                    )
            
            restricted_correction = scipy.sparse.linalg.spsolve(stiffness, restricted_residual)
            
            ext_corr = self.extend_partition_of_unity_RAS(  part, 
                                                            restricted_correction, 
                                                            self.partition.nodes_membership_overlap, 
                                                            dict_v2d_map, 
                                                            self.partition.dict_restriction)
            
            extended_correction += ext_corr

        if level == 1 :         
            full_solution = extended_correction
        elif level == 2 : 
            full_solution = extended_correction + coarse_solution

        return full_solution

    #########################################
    ## LOW LEVEL METHODS FOR 2-LEVEL DDM ####
    #########################################

    def extract_global_problem(self):
        '''
        Returns the stiffness matrix and vertex to dof map 
        of a Poisson problem computed with Fenics on the full mesh. 
        '''
        
        V = FunctionSpace(self.full_mesh, "Lagrange", 1)

        u = TrialFunction(V)
        v = TestFunction(V)

        bc = DirichletBC(V, self.g_expression, self.full_facet, 101)

        a = inner(grad(u), grad(v))*dx
        L = self.f_expression*v*dx

        A = assemble(a)
        rhs = assemble(L)

        bc.apply(A, rhs)

        global_A = A.copy().array()
        global_rhs = rhs.get_local().reshape(-1,1)
        global_d2v = dof_to_vertex_map(V)
        global_v2d = vertex_to_dof_map(V)
        
        return global_A, global_rhs, global_d2v, global_v2d

    def compute_nicolaides_operator(self):
        '''
        Returns the Nicolaide coarse matrix of size (NxNpart).
        The matrix is mesh-ordered.
        '''
        
        membership = self.partition.original_membership
        membership_overlap = self.partition.nodes_membership_overlap

        index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]

        diff = list(set(list(membership)))
        
        Z = np.zeros((len(membership), len(diff)))
        
        for i in range(len(membership_overlap)) :
            sublist = membership_overlap[i]
            for k in sublist :
                Z[i,k] = 1/len(sublist)

        Z[index,:] = 0

        return Z
    
    def compute_coarse_operators(self, A, Z):
        """
        Returns the coarse matrix Q = Z^TAZ and its inverse
        E = Q^-1. Z must follow A ordering.
        """
        Q = np.dot(Z.T, np.dot(A, Z))
        E = scipy.linalg.inv(Q)

        return Q, E

    def compute_coarse_correction(self, residual, Z, Qinv, global_d2v, global_v2d):
        """
        Returns the coarse correction mapped back
        to the full problem size and ordered in vertex
        order.
        """

        #convert residual to dof order
        residual = residual.copy()[global_d2v]
        #restrict residual to coarse space
        restricted_residual = np.dot(Z.T, residual.reshape(-1,1))
        #multiply by the inverse operator
        sol_restricted_residual = np.dot(Qinv, restricted_residual)
        #map back to size of global problem, flatten and re-order
        coarse_correction = np.dot(Z, sol_restricted_residual)
        coarse_correction = (coarse_correction.flatten())[global_v2d]

        return coarse_correction 
    
    #########################################
    ########## KRYLOV SOLVER ################
    #########################################
    
    def cg(self, dict_global, x0, stop_mode = "norm", tol = 1e-6, max_iter = 1000):

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
        p = r.copy()

        rho_prev = 1

        for iter in tqdm(range(max_iter)):

            rho_prev = np.dot(r, r)

            q = (A @ p.reshape(-1,1)).flatten()

            alpha = rho_prev / np.dot(p, q)
            
            x = x + alpha * p
            r = r - alpha * q

            store_sol[str(iter)] = (x.copy())[global_v2d_map] 

            if stop_mode == "norm":
                store_residual[str(iter)] = np.linalg.norm(r)
            elif stop_mode == "mse":
                store_residual[str(iter)] = np.mean(r**2)

            if store_residual[str(iter)] < tol :
                break;

            rho = np.dot(r, r)
            
            beta = rho / rho_prev

            p = r + beta*p
            
            rho_prev = rho

        return store_sol, store_residual

    def cg_preconditioner(self, dict_global, dict_preconditioner, x0, level = 1, stop_mode = "norm", tol = 1e-6, max_iter = 1000):

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
        z = self.ASM_preconditioner(r.copy()[global_v2d_map], dict_global, dict_preconditioner, level = level)
        z = z[global_d2v_map]

        p = z.copy()

        rho_prev = 1

        for iter in tqdm(range(max_iter)):

            rho_prev = np.dot(r, z)

            q = (A @ p.reshape(-1,1)).flatten()

            alpha = rho_prev / np.dot(p, q)
            
            x = x + alpha * p
            r = r - alpha * q

            store_sol[str(iter)] = (x.copy())[global_v2d_map] 

            if stop_mode == "norm":
                store_residual[str(iter)] = np.linalg.norm(r)
            elif stop_mode == "mse":
                store_residual[str(iter)] = np.mean(r**2)

            if store_residual[str(iter)] < tol :
                break;

            z = self.ASM_preconditioner(r.copy()[global_v2d_map], dict_global, dict_preconditioner, level = level)
            z = z[global_d2v_map]

            rho = np.dot(r, z)
            
            beta = rho / rho_prev

            p = z + beta*p
            
            rho_prev = rho

        return store_sol, store_residual

    def bicgstab(self, dict_global, x0, stop_mode = "norm", tol = 1e-6, max_iter = 1000):

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
        
        r_tilde = r.copy()

        rho_prev, alpha, omega = 1, 1, 1

        for iter in tqdm(range(max_iter)):

            rho = np.dot(r_tilde, r)

            # if orthogonal then it fails
            if rho == 0:
                sys.exit("Method Fails")
            
            if iter == 0 :
                p = r.copy()
            
            else : 
                beta = (rho / rho_prev) * (alpha / omega)
                p = r + beta * (p - omega * v)

            v = (A @ p.reshape(-1,1)).flatten()

            # compute alpha
            alpha = rho / np.dot(r_tilde, v)
            # compute new residual s
            s = r - alpha * v

            if stop_mode == "norm":
                residual = np.linalg.norm(s)
                if residual < tol : 
                    x = x + alpha * p
                    store_sol[str(iter)] = (x.copy())[global_v2d_map] 
                    store_residual[str(iter)] = residual
                    break;
            elif stop_mode == "mse":
                residual = np.mean(s**2)
                if residual < tol : 
                    x = x + alpha * p
                    store_sol[str(iter)] = (x.copy())[global_v2d_map] 
                    store_residual[str(iter)] = residual
                    break;
            
            t = (A @ s.reshape(-1,1)).flatten()

            # compute omega
            omega = np.dot(t, s) / np.dot(t, t)

            # compute next x
            x = x + alpha * p + omega * s

            # compute residual for next step
            r = s - omega * t

            store_sol[str(iter)] = (x.copy())[global_v2d_map] 

            if stop_mode == "norm":
                store_residual[str(iter)] = np.linalg.norm(r)
            elif stop_mode == "mse":
                store_residual[str(iter)] = np.mean(r**2)

            if store_residual[str(iter)] < tol :
                break;
                        
            rho_prev = rho

        return store_sol, store_residual
    
    def bicgstab_preconditioner(self, dict_global, dict_preconditioner, x0, level = 1, stop_mode = "norm", tol = 1e-6, max_iter = 1000):

        # initialise saving dictionnaries        
        store_sol = {}
        store_residual = {}

        # extract data
        A = dict_global["global_A"]
        rhs = dict_global["global_rhs"]
        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]
           
        # initial guess
        x = (x0.copy())[global_d2v_map]

        # initial residual
        r = (rhs - A @ x.reshape(-1,1)).flatten()

        r_tilde = r.copy()

        rho_prev, alpha, omega = 1, 1, 1

        for iter in tqdm(range(max_iter)):

            # compute rho
            rho = np.dot(r_tilde, r)
            # if orthogonal then it fails
            if rho == 0:
                sys.exit("Method Fails")
            
            if iter == 0 :
                p = r.copy()
            
            else : 
                beta = (rho / rho_prev) * (alpha / omega)
                p = r + beta * (p - omega * v)

            p_hat = self.RAS_preconditioner(p.copy()[global_v2d_map], dict_global, dict_preconditioner, level = level)
            p_hat = p_hat[global_d2v_map]

            v = (A @ p_hat.reshape(-1,1)).flatten()

            # compute alpha
            alpha = rho / np.dot(r_tilde, v)
            # compute new residual s
            s = r - alpha * v

            if stop_mode == "norm":
                residual = np.linalg.norm(s)
                if residual < tol : 
                    x = x + alpha * p_hat
                    store_sol[str(iter)] = (x.copy())[global_v2d_map] 
                    store_residual[str(iter)] = residual
                    break;
            elif stop_mode == "mse":
                residual = np.mean(s**2)
                if residual < tol : 
                    x = x + alpha * p_hat
                    store_sol[str(iter)] = (x.copy())[global_v2d_map] 
                    store_residual[str(iter)] = residual
                    break;

            # solve second RAS
            s_hat = self.RAS_preconditioner(s.copy()[global_v2d_map], dict_global, dict_preconditioner, level = level)
            s_hat = s_hat[global_d2v_map]

            t = (A @ s_hat.reshape(-1,1)).flatten()

            # compute omega
            omega = np.dot(t, s) / np.dot(t, t)

            # compute next x
            x = x + alpha * p_hat + omega * s_hat

            # compute residual for next step
            r = s - omega * t

            store_sol[str(iter)] = (x.copy())[global_v2d_map] 

            if stop_mode == "norm":
                store_residual[str(iter)] = np.linalg.norm(r)
            elif stop_mode == "mse":
                store_residual[str(iter)] = np.mean(r**2)

            if store_residual[str(iter)] < tol :
                break;

        return store_sol, store_residual

