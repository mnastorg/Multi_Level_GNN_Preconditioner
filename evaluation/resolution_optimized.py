### PACKAGES ###
################

import os
import sys 
import time 
import warnings
from tqdm import tqdm 
import json 
from json import JSONEncoder

import scipy
import numpy as np
from copy import deepcopy
from scipy.sparse import find, csr_matrix, SparseEfficiencyWarning

from fenics import *

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import model_dss_inference as modtest

warnings.simplefilter('ignore', SparseEfficiencyWarning)

################
################

class DDMResolution():

    def __init__(self, dict_resolution):

        self.partition = dict_resolution["partition"]
        self.path_full_mesh = dict_resolution["path_full_mesh_hdf5"]
        self.path_subdomains = dict_resolution["path_subdomains"]

        self.path_savings = dict_resolution["path_savings"]
        self.path_ml_model = dict_resolution["path_model"]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.batch_size = self.partition.Nparts

        np.random.seed(dict_resolution["seed"])

        if self.path_ml_model is not None :             
            checkpoint = torch.load(self.path_ml_model)
            config_model = checkpoint["hyperparameters"]
            self.model = modtest.ModelDSS(config_model)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device)
            # print("Nb Parameters : ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        else : 
            self.model = None 

        # PDDM PrbData Original Normalized Large
        # self.prb_data_mean = torch.tensor([0.0], dtype = torch.float)
        # self.prb_data_std = torch.tensor([0.013], dtype = torch.float)

        self.prb_data_mean = torch.tensor([0.0], dtype = torch.float)
        self.prb_data_std = torch.tensor([0.0129], dtype = torch.float)

        # force function and boundary value
        self.param_f = np.random.uniform(-10, 10, 3)
        print("Force function : ", self.param_f)
        self.param_g = np.random.uniform(-10, 10, 6)
        print("Boundary function : ", self.param_g)

        # define expressions of functions f and g
        L = dict_resolution["radius"]

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
        
        self.f_array = self.f_expression.compute_vertex_values(self.full_mesh) 
        self.g_array = self.g_expression.compute_vertex_values(self.full_mesh)    
    
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
        
        start = time.time()
        solve(A, u.vector(), rhs, "lu")
        end = time.time()
        total_time = end - start

        sol = u.compute_vertex_values(self.full_mesh)

        return sol, total_time
    
    def solve_cg(self, preconditioner = None, level = 1, stop_mode = "norm", x0 = None, tol = 1.e-6, max_iter = 500):
        """
        Solve the linear system arising from discretization of a Poisson problem
        using conjugate gradient.

        - preconditioner = None     :   no preconditioner
                         = "ASM-LU" :   ASM preconditioner
                         = "ASM-GNN":   ASM preconditioner with GNN 
                         = "ICC"    :   Incomplete Cholesky

        if preconditioner is not None, level = 1 or level = 2 for resp. the 
        classical 1-level method or coarse space correction (2-level).

        - stop_mode = "norm" to stop at the norm of the residual
                    = "mse" to stop at the MSE of the residual.

        Outputs dictionnary of solution and residual at each iteration.
        until convergence.        
        """

        if (preconditioner != None) & (preconditioner != "ASM-LU") & (preconditioner != "ASM-GNN") & (preconditioner != "ICC"):
            sys.exit("select other preconditioner")

        if (stop_mode != 'norm') & (stop_mode != 'mse') & (stop_mode != 'res_rel') :
            sys.exit("stop_mode should be norm or mse")

        if (level != 1) & (level != 2) :
            sys.exit("level should be 1 or 2.")

        # Extract global data
        global_A, global_rhs, global_d2v_map, global_v2d_map = self.extract_global_problem()

        dict_global = {"global_A" : global_A,
                       "global_rhs" : global_rhs,
                       "global_d2v_map" : global_d2v_map,
                       "global_v2d_map" : global_v2d_map
                       }

        # Extract preconditioner data if required
        if preconditioner is not None : 

            dict_A, dict_rhs, dict_v2d_map, dict_d2v_map = self.extract_fenics_data_subdomains()
            
            dict_stiffness_with_bc = self.dict_stiffness_with_bc(   dict_A,
                                                                    dict_d2v_map,
                                                                    self.partition.dict_restriction,
                                                                    self.partition.dict_is_interface)
            
            dict_preconditioner = { "dict_stiffness_with_bc" : dict_stiffness_with_bc,
                                    "dict_rhs" : dict_rhs,
                                    "dict_d2v_map" : dict_d2v_map,
                                    "dict_v2d_map" : dict_v2d_map
                                    }
            
            # If level == 2 then extract coarse space data
            if level == 2 :

                Z = self.compute_nicolaides_operator()
                Z = Z[global_d2v_map]
                Q, Qinv = self.compute_coarse_operators(global_A, Z)

                dict_preconditioner["nicolaides_operator"] = Z
                dict_preconditioner["inv_coarse_operator"] = Qinv

        # If no initial solution then provide one
        if x0 == None :

            global_sol = np.zeros(self.partition.full_graph.number_of_nodes())
            index = np.where(self.partition.full_graph.domain_boundary_tags == 101)[0]
            global_sol[index] = self.g_array[index]

        else : 

            global_sol = x0

        # If no preconditioner is required
        if preconditioner == None :
            
            solution, residual, inference_time = self.conjugate_gradient(   global_sol,
                                                                            dict_global, 
                                                                            dict_preconditioner=None,
                                                                            prec_type=None,
                                                                            stop_mode = stop_mode, 
                                                                            tol = tol, 
                                                                            max_iter = max_iter)
                                                
        # If ASM-LU preconditioner
        if preconditioner == "ASM-LU":

            solution, residual, inference_time = self.conjugate_gradient(   global_sol,
                                                                            dict_global, 
                                                                            dict_preconditioner=dict_preconditioner,
                                                                            prec_type="ASM-LU",
                                                                            level = level,
                                                                            stop_mode = stop_mode, 
                                                                            tol = tol, 
                                                                            max_iter = max_iter)

        # If ASM-GNN preconditioner  
        if preconditioner == "ASM-GNN":

            solution, residual, inference_time = self.conjugate_gradient(   global_sol,
                                                                            dict_global, 
                                                                            dict_preconditioner=dict_preconditioner,
                                                                            prec_type="ASM-GNN",
                                                                            level = level,
                                                                            stop_mode = stop_mode, 
                                                                            tol = tol, 
                                                                            max_iter = max_iter)

        # If ICC preconditioner  
        if preconditioner == "ICC":
            
            solution, residual, inference_time = self.conjugate_gradient(   global_sol,
                                                                            dict_global, 
                                                                            dict_preconditioner=dict_preconditioner,
                                                                            prec_type="ICC",
                                                                            stop_mode = stop_mode, 
                                                                            tol = tol, 
                                                                            max_iter = max_iter)

        return solution, residual, inference_time

    ###############################################
    ############# CONJUGATE GRADIENT ##############
    ###############################################

    def conjugate_gradient(self, x0, dict_global, dict_preconditioner = None, prec_type = "ASM", level = 1, stop_mode = "norm", tol = 1e-6, max_iter = 1000):

        start = time.time()

        # store solution
        store_sol, store_residual = {}, {}
        total_time, iteration_time, inference_time = [], [], []

        # extract data
        global_A = dict_global["global_A"]
        global_rhs = dict_global["global_rhs"]
        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]

        # initial guess
        x = (x0.copy())[global_d2v_map]
        store_sol["0"] = (x.copy())[global_v2d_map] 

        # initial residual
        r = (global_rhs - global_A @ x.reshape(-1,1)).flatten()

        if stop_mode == "norm":
            store_residual["0"] = np.linalg.norm(r)
        if stop_mode == "res_rel":
            store_residual["0"] = np.linalg.norm(r) / np.linalg.norm(global_rhs)
        elif stop_mode == "mse":
            store_residual["0"] = np.mean(r**2)
 
        if dict_preconditioner is not None :

            if prec_type == "ASM-LU" : 

                z, infer_time = self.ASM_LU(    r.copy()[global_v2d_map], 
                                                dict_global, 
                                                dict_preconditioner, 
                                                level = level)
                z = z[global_d2v_map]

                inference_time.append(infer_time)

            elif prec_type == "ASM-GNN":
                
                list_data_subdomains = self.construct_original_batch(dict_preconditioner["dict_stiffness_with_bc"], 
                                                                     dict_preconditioner["dict_d2v_map"])
                
                z, infer_time = self.ASM_GNN(   r.copy()[global_v2d_map], 
                                                dict_global, 
                                                dict_preconditioner,
                                                list_data_subdomains,
                                                iter_id=0,
                                                level = level)

                z = z[global_d2v_map]
                
                inference_time.append(infer_time)

            elif prec_type == "ICC":
                
                L = self.incomplete_cholesky(global_A)

                y = scipy.linalg.solve_triangular(L, r.copy(), lower = True)
                z = scipy.linalg.solve_triangular(L.T, y, lower = False)

            p = z.copy()

        else : 

            p = r.copy()

        rho_prev = 1
        
        for iter in tqdm(range(1,max_iter)):
            
            start_iter_time = time.time()

            if dict_preconditioner is None : 

                rho_prev = np.dot(r, r)

                q = (global_A @ p.reshape(-1,1)).flatten()

                alpha = rho_prev / np.dot(p, q)
                
                x = x + alpha * p
                r = r - alpha * q
                
                store_sol[str(iter)] = (x.copy())[global_v2d_map] 
                
                if stop_mode == "norm":
                    store_residual[str(iter)] = np.linalg.norm(r)
                if stop_mode == "res_rel":
                    store_residual[str(iter)] = np.linalg.norm(r) / np.linalg.norm(global_rhs)
                elif stop_mode == "mse":
                    store_residual[str(iter)] = np.mean(r**2)
                    
                if store_residual[str(iter)] < tol :
                    break;

                rho = np.dot(r,r)

                beta = rho / rho_prev

                p = r + beta*p
                
                rho_prev = rho

            else :

                rho_prev = np.dot(r, z)

                q = (global_A @ p.reshape(-1,1)).flatten()

                alpha = rho_prev / np.dot(p, q)
                
                x = x + alpha * p
                r = r - alpha * q

                store_sol[str(iter)] = (x.copy())[global_v2d_map] 
                
                if stop_mode == "norm":
                    store_residual[str(iter)] = np.linalg.norm(r)
                if stop_mode == "res_rel":
                    store_residual[str(iter)] = np.linalg.norm(r) / np.linalg.norm(global_rhs)
                elif stop_mode == "mse":
                    store_residual[str(iter)] = np.mean(r**2)
                    
                if store_residual[str(iter)] < tol :
                    break;

                if prec_type == "ASM-LU" : 

                    z, infer_time = self.ASM_LU(    r.copy()[global_v2d_map], 
                                                    dict_global, 
                                                    dict_preconditioner, 
                                                    level = level)
                    
                    z = z[global_d2v_map]

                    inference_time.append(infer_time)
                
                elif prec_type == "ASM-GNN":
                    
                    z, infer_time = self.ASM_GNN(   r.copy()[global_v2d_map], 
                                                    dict_global, 
                                                    dict_preconditioner,
                                                    list_data_subdomains,
                                                    iter_id=iter, 
                                                    level = level)

                    z = z[global_d2v_map]

                    inference_time.append(infer_time)

                elif prec_type == "ICC":
                    
                    y = scipy.linalg.solve_triangular(L, r.copy(), lower = True)
                    z = scipy.linalg.solve_triangular(L.T, y, lower = False)

                rho = np.dot(r, z)
                
                beta = rho / rho_prev

                p = z + beta*p
                
                rho_prev = rho

            end_iter_time = time.time()

            iteration_time.append(end_iter_time - start_iter_time)
        
        end = time.time()

        total_time.append(end - start)

        return store_sol, store_residual, inference_time

    ###############################################
    ################ PRECONDITIONERS ##############
    ###############################################

    def incomplete_cholesky(self, a):

        n = a.shape[0]
        L = np.zeros_like(a)

        for k in range(n): 
            L[k,k] = np.sqrt(a[k,k])
            i_ = (a[k+1:,k].nonzero())[0]
            if len(i_) > 0:
                i_= i_ + (k+1)
                L[i_,k] = a[i_,k]/a[k,k]
            for j in i_: 
                i2_ = (a[j:n,j].nonzero())[0]
                if len(i2_) > 0:
                    i2_ = i2_ + j
                    L[i2_,j]  = a[i2_,j] - a[i2_,k]*a[j,k]   

        return L    

    def ASM_LU(self, rhs, dict_global, dict_preconditioner, level = 1):
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

        cumultime = []
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
            infer_start = time.time()
            restricted_correction = scipy.sparse.linalg.spsolve(stiffness, restricted_residual)
            infer_end = time.time()
            cumultime.append(infer_end - infer_start)

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
        
        return full_solution, sum(cumultime)

    def ASM_GNN(self, rhs, dict_global, dict_preconditioner, list_data_subdomains, iter_id, level = 1):

        global_d2v_map = dict_global["global_d2v_map"]
        global_v2d_map = dict_global["global_v2d_map"]

        dict_d2v_map = dict_preconditioner["dict_d2v_map"]
        dict_v2d_map = dict_preconditioner["dict_v2d_map"]

        if level == 2 :
            Z = dict_preconditioner["nicolaides_operator"]
            Qinv = dict_preconditioner["inv_coarse_operator"]
            coarse_solution = self.compute_coarse_correction(rhs, Z, Qinv, global_d2v_map, global_v2d_map)
        
        rhs_norm = np.linalg.norm(rhs)

        self.update_batch(list_data_subdomains, rhs/rhs_norm, dict_d2v_map)

        loader = DataLoader(list_data_subdomains, batch_size = self.batch_size, shuffle = False)
        
        infer_start = time.time()
        batch_solution = self.compute_ml_solution(loader)
        infer_time = time.time() - infer_start

        extended_correction = self.extend_batch_with_partition_of_unity(list_data_subdomains, batch_solution, dict_v2d_map)

        extended_correction = extended_correction * rhs_norm

        if level == 1 :         

            full_solution = extended_correction

        elif level == 2 : 

            full_solution = extended_correction + coarse_solution
        
        return full_solution, infer_time
        
    ###############################################
    # LOW LEVEL METHODS FOR 1 AND 2 LEVEL SCHWARZ #
    ###############################################

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

        # A = assemble(a)
        # rhs = assemble(L)

        # bc.apply(A, rhs)

        A, rhs = assemble_system(a, L, bc)

        global_A = A.copy().array()
        global_rhs = rhs.get_local().reshape(-1,1)
        global_d2v = dof_to_vertex_map(V)
        global_v2d = vertex_to_dof_map(V)
        
        return global_A, global_rhs, global_d2v, global_v2d

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
    ####### Machine Learning Krylov #########
    #########################################

    def construct_batch(self, batch_config):
        """From a dictionnary of inputs, extract a batch of subproblems 
        to be run in the ML model. Return a list of data objects."""
        
        data_list = []

        dict_stiffness = batch_config["dict_stiffness"]
        residual_vector = batch_config["dict_rhs"]
        dict_d2v_map = batch_config["dict_d2v_map"]
        # dict_v2d_map = batch_config["dict_v2d_map"]
        # global_sol = batch_config["global_sol"]
        l_res = []

        ## extract data
        for part in range(self.partition.Nparts):

            d2v = dict_d2v_map[str(part)]
            
            stiffness = dict_stiffness[str(part)]
            dense_stiffness = stiffness.todense()
            
            restricted_residual = self.compute_restricted_residual(part,
                                                                   residual_vector,
                                                                   self.partition.full_graph, 
                                                                   dict_d2v_map, 
                                                                   self.partition.dict_restriction, 
                                                                   self.partition.dict_is_interface
                                                                   )
            
            # restricted_residual = restricted_residual/np.linalg.norm(restricted_residual)
            l_res.append(restricted_residual)

            sub_sol = scipy.linalg.solve(dense_stiffness, restricted_residual.reshape(-1,1))

            restriction = self.partition.dict_restriction[str(part)]
            interface = self.partition.dict_is_interface[str(part)] 

            prb_data = np.zeros_like(restriction).reshape(-1,1)
            tags = np.zeros_like(restriction).reshape(-1,1)

            sub_dof_full_boundary = (self.partition.full_graph.domain_boundary_tags[restriction])[d2v]
            sub_dof_interface = (interface[restriction])[d2v]
            sub_dof_coordinates = (self.partition.full_graph.points[restriction][d2v])

            index_tags_boundary = np.where((sub_dof_full_boundary == 101) | (sub_dof_interface == True))[0]
            tags[index_tags_boundary,:] = 1

            ## build pytorch tensors
            coefficients = np.asarray(scipy.sparse.find(stiffness))
            
            edge_index = torch.tensor(coefficients[:2,:].astype('int'), dtype=torch.long)
            
            a_ij = torch.tensor(coefficients[2,:].reshape(-1,1), dtype=torch.float)
            b_tensor = torch.tensor(restricted_residual.reshape(-1,1), dtype=torch.float)

            sol_tensor = torch.tensor(sub_sol.reshape(-1,1), dtype = torch.float)

            # Extract prb_data
            prb_data_tensor = torch.tensor(prb_data, dtype = torch.float)
            restricted_residual_tensor = torch.tensor(restricted_residual, dtype=torch.float)
            prb_data_tensor[:,0] = restricted_residual_tensor
            prb_data_tensor = (prb_data_tensor - self.prb_data_mean) / self.prb_data_std

            # Extract tags to differentiate nodes 
            tags_tensor = torch.tensor(tags, dtype=torch.float)

            # Extract coordinates
            pos_tensor = torch.tensor(sub_dof_coordinates, dtype = torch.float)

            # Extract initial condition
            x_tensor = torch.zeros_like(sol_tensor)

            data = Data(x = x_tensor, edge_index = edge_index, 
                        a_ij = a_ij, y = b_tensor, 
                        sol = sol_tensor, prb_data = prb_data_tensor, tags = tags_tensor,  
                        pos = pos_tensor
                        )
            
            data_list.append(data)

        transform = T.Compose([T.Cartesian(), T.Distance()])
        data_list = [transform(data) for data in data_list]

        return data_list, l_res

    def construct_original_batch(self, dict_stiffness, dict_d2v_map):
        """From a dictionnary of inputs, extract a batch of subproblems 
        to be run in the ML model. Return a list of data objects."""
        
        data_list = []

        for part in range(self.partition.Nparts):

            d2v = dict_d2v_map[str(part)]
            
            restriction = self.partition.dict_restriction[str(part)]
            interface = self.partition.dict_is_interface[str(part)] 

            # prb_data = np.zeros((len(restriction),3))
            prb_data = np.zeros_like(restriction).reshape(-1,1)
            tags = np.zeros_like(restriction).reshape(-1,1)

            sub_dof_full_boundary = (self.partition.full_graph.domain_boundary_tags[restriction])[d2v]
            sub_dof_interface = (interface[restriction])[d2v]
            sub_dof_coordinates = (self.partition.full_graph.points[restriction][d2v])

            index_tags_boundary = np.where((sub_dof_full_boundary == 101) | (sub_dof_interface == True))[0]
            tags[index_tags_boundary,:] = 1

            ## build pytorch tensors
            stiffness = dict_stiffness[str(part)]
            coefficients = np.asarray(scipy.sparse.find(stiffness))
            edge_index = torch.tensor(coefficients[:2,:].astype('int'), dtype=torch.long)

            # Extract prb_data
            prb_data_tensor = torch.tensor(prb_data, dtype = torch.float)

            # Extract tags to differentiate nodes 
            tags_tensor = torch.tensor(tags, dtype=torch.float)

            # Extract coordinates
            pos_tensor = torch.tensor(sub_dof_coordinates, dtype = torch.float)

            # Extract initial condition
            x_tensor = torch.zeros_like(tags_tensor)

            data = Data(x = x_tensor, edge_index = edge_index,
                        prb_data = prb_data_tensor, tags = tags_tensor,  
                        pos = pos_tensor)
            
            data_list.append(data)

        transform = T.Compose([T.Cartesian(), T.Distance()])
        data_list = [transform(data) for data in data_list]

        return data_list
       
    def update_batch(self, list_data_subdomains, residual_vector, dict_d2v_map):
    
        ## extract data
        for part in range(self.partition.Nparts):

            data = list_data_subdomains[part]
            
            restricted_residual = self.compute_restricted_residual(part,
                                                                   residual_vector,
                                                                   self.partition.full_graph, 
                                                                   dict_d2v_map, 
                                                                   self.partition.dict_restriction, 
                                                                   self.partition.dict_is_interface
                                                                   )

            restricted_residual = torch.tensor(restricted_residual, dtype=torch.float)

            data.prb_data[:,0] = restricted_residual
            data.prb_data = (data.prb_data - self.prb_data_mean) / self.prb_data_std

    def extend_batch_with_partition_of_unity(self, data_list, solution, dict_v2d_map):

        cumul = 0

        extended_correction = np.zeros(self.partition.full_graph.number_of_nodes())
        
        for part in range(self.partition.Nparts):
        
            d = data_list[part]
            length = len(d.x)

            ml_solution = solution[cumul:cumul + length].cpu().numpy().flatten()
            boundary_index = np.where(d.tags == 1)[0]
            ml_solution[boundary_index] = 0

            ext_corr = self.extend_partition_of_unity_ASM(  part, 
                                                            ml_solution, 
                                                            self.partition.nodes_membership_overlap, 
                                                            dict_v2d_map, 
                                                            self.partition.dict_restriction)
            
            cumul += length
            
            extended_correction = extended_correction + ext_corr

        return extended_correction
    
    def compute_ml_solution(self, loader):
        
        self.model.eval()
        
        sol = []        
        
        with torch.no_grad() :

            for i, test_data in enumerate(loader) :
                
                U_sol = self.model(test_data.to(self.device))
                
                sol.append(U_sol)

        return torch.vstack(sol)

    def save_sparse_global_problem(self):
        '''
        Save data from a Poisson problem 
            - Delta(u)  = f in Omega
                    u   = g in boundaryOmega 
        computed with Fenics on the full mesh 
        '''

        V = FunctionSpace(self.full_mesh, "Lagrange", 1)

        d2v = dof_to_vertex_map(V)
        vertex_coordinates = self.full_mesh.coordinates()
        dof_coordinates = vertex_coordinates[d2v]

        u = TrialFunction(V)
        v = TestFunction(V)

        bc = DirichletBC(V, self.g_expression, self.full_facet, 101)

        a = inner(grad(u), grad(v))*dx
        L = self.f_expression*v*dx

        A = assemble(a)
        rhs = assemble(L)

        bc.apply(A, rhs)

        u = Function(V)
        
        A_mat = as_backend_type(A).mat()
        A_sparse_matrix = csr_matrix(A_mat.getValuesCSR()[::-1], shape = A_mat.size)
        b_matrix = rhs.get_local().reshape(-1, 1)

        tags = np.zeros_like(b_matrix)
        g_boundary = bc.get_boundary_values()
        g_boundary = list(g_boundary.items())
        for items in g_boundary:
            tags[items[0]] = 1

        np.save(os.path.join(self.path_savings, "A_sparse_matrix.npy"), [A_sparse_matrix], allow_pickle=True)
        np.save(os.path.join(self.path_savings, "b_matrix.npy"), [b_matrix], allow_pickle=True)
        np.save(os.path.join(self.path_savings, "tags.npy"), [tags], allow_pickle = True)
        np.save(os.path.join(self.path_savings, "coordinates.npy"), [dof_coordinates], allow_pickle = True)
