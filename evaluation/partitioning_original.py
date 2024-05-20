### PACKAGES ###
import os
import numpy as np
import networkx as nx 
import pymetis as metis 
from copy import deepcopy
import meshio 
import shutil 
from tqdm import tqdm
################

class Partitioning():

    def __init__(self, part_dic):

        # meshio to read the mesh
        self.path_mesh = part_dic["path_mesh_xdmf"]
        self.path_save_subdomains = part_dic["path_save_subdomains"]
        self.mesh = meshio.read(self.path_mesh)
        self.points = self.mesh.points[:,:2]
        self.cells = self.mesh.cells_dict['triangle']
        self.edges = self.build_edges_from_cells()

        # define some parameters
        self.Nparts = part_dic["Nparts"] 
        self.Noverlaps = part_dic["Noverlaps"]

        # initialize the full graph
        self.full_graph = self.build_full_graph()

    def framework_partitioning(self):
        '''
        Full pipeline to extract and save all data for 
        DDM resolution with Fenics.
        '''

        print('--> Build original partitioning into {} subdomains'.format(self.Nparts)) 
        self.original_membership = self.partitioning()
        self.full_graph.membership = self.original_membership
        print('--> Compute dict of {} subdomains with an overlap of {}'.format(self.Nparts, self.Noverlaps))
        self.dict_subdomains = self.dict_overlapping_subdomains()
        print('--> Compute in which membership each node lives (can be several)')
        self.nodes_membership_overlap = self.check_membership(self.dict_subdomains)
        print('--> Compute dict for the cells of the mesh, relative to the overlapping subdomains')
        self.dict_cells = self.dict_overlapping_cells(self.dict_subdomains)
        print("--> Compute dict of restriction arrays for each subdomain")
        self.dict_restriction = self.dict_restriction_map(self.dict_subdomains)
        print("--> Compute dict of interface boolean arrays for each subdomain")
        self.dict_is_interface = self.dict_boolean_interface(self.dict_subdomains)
        print('--> Saving all subdomains in {}'.format(self.path_save_subdomains))
        self.save_subdomains_to_xdmf(self.dict_subdomains, self.dict_cells)

    def build_edges_from_cells(self):
        ''' 
        Function to build edges from cells of the mesh.
        Return a list of tuples (e0, e1) for each edge of the
        mesh.
        '''
        edges = []
        for i in range(len(self.cells)): 
            t = self.cells[i,:]
            edges.append((t[0],t[1]))
            edges.append((t[0],t[2]))
            edges.append((t[1],t[2]))        

        return edges
    
    def build_full_graph(self):
        ''' 
        Function to build the full graph of the mesh
        import from path_mesh. The graph is a NetworkX graph
        object with all the nodes ordered.
        '''
        
        # construct the NetworkX Graph 
        G = nx.Graph()
        G.add_edges_from(self.edges)
        G.points = self.points
        # warnings : the nodes of G are not ordered.
        # now construct another Graph H with ordered nodes
        H = nx.Graph()
        H.add_nodes_from(sorted(G.nodes(data=True)))
        H.add_edges_from(G.edges(data=True))
        H.points = self.points 
        
        # construct boundary tags
        lines_boundary = self.mesh.cells_dict["line"]
        domain_boundary_tags = np.zeros(np.shape(self.points)[0])
        index_boundary = np.asarray(list(set(lines_boundary.flatten().tolist())))
        # array of size Nnodes with 0 if interior and 101 if dirichlet
        self.dirichlet_tag = 101
        domain_boundary_tags[index_boundary] = self.dirichlet_tag
        H.domain_boundary_tags = domain_boundary_tags
        return H
    
    def partitioning(self):
        '''
        Returns an array of size Nnodes, indicating in which partition
        each node live.
        '''

        _, membership = metis.part_graph(self.Nparts, adjacency=self.full_graph.adj)

        return np.asarray(membership)

    def node_at_interface(self, graph, node, membership):
        '''
        Computes if a node (node) in graph (graph) is at the interface with another partition.
        Output : (bool, neighbors_index, neighbors_membership) 
        - bool : False  if node is interior to its partition and return empty list.
        - bool : True   if node is a the interface and return its neighbors' memberships (tag : np.array)
        '''
        
        #membership of node
        membership_node = membership[node]

        #list of neighbors of node 
        neighbors_index = np.asarray([n for n in graph.neighbors(node)])
        
        #list of membership of node's neighbors
        list_neighbors_membership = membership[neighbors_index]
        
        #boolean if all neighbors have the same membership as node
        bool = not np.all(list_neighbors_membership == membership_node)

        #if true then node is on the boundary and return its neighbors'membership
        if bool :
            index = np.where(list_neighbors_membership != membership_node)[0]
            neighbors_membership = np.unique(list_neighbors_membership[index])
        #if false then node is interior and return empty list
        else :
            neighbors_membership = np.array([])

        return bool, neighbors_index.tolist(), neighbors_membership.tolist()

    def interface(self, graph, membership):
        '''
        Computes informative arrays for all nodes in the full graph.
        Graph is a NetworkX object and membership an array of size Nnodes in
        the graph.
        For a node i :
            - numpy array : True (boolean) if at interface between subdomains.
            - list of the index of its neighbors.
            - list of its neighboring subdomains (empty if interior).
        '''
        
        is_interface, neighbors_index, neighbors_membership = [], [], []
        
        for node in range(graph.number_of_nodes()):
            b, i, t = self.node_at_interface(graph, node, membership)
            is_interface.append(b)
            neighbors_index.append(i)
            neighbors_membership.append(t)

        return np.asarray(is_interface), neighbors_index, neighbors_membership

    def create_subgraph(self, graph, id):
        '''
        Returns a NetworkX subgraph of object graph 
        corresponding to the subdomain id.
        '''
        arg = np.where(graph.membership == id)[0]
        subgraph = graph.subgraph(arg)
        subgraph.membership = graph.membership[arg]
        subgraph.is_interface = graph.is_interface[arg]
        subgraph.neighbors_index = [graph.neighbors_index[k] for k in arg]
        subgraph.neighbors_membership = [graph.neighbors_membership[k] for k in arg]
        
        return subgraph
    
    def dict_overlapping_subdomains(self):
        '''Returns a dictionnary of membership arrays with an overlap of Noverlap for 
        each subdomain i in Npart.
        Example : dict = {  '0' : membership_array_subdomain_0, 
                            '1' : membership_array_subdomain_1}
        '''
        dict = {}
        for part in tqdm(range(self.Nparts)):
            membership = self.build_overlapping_subdomain(part, overlap=self.Noverlaps)
            dict[str(part)] = membership 
        
        return dict
    
    def build_overlapping_subdomain(self, id, overlap = 1):
        '''
        For one subdomain (id), returns the new membership array that corresponds to 
        an overlapping of overlap.
        '''
        new_graph = deepcopy(self.full_graph)   
        for nlaps in range(overlap):
            is_interface, neighbors_index, neighbors_membership = self.interface(new_graph, new_graph.membership)
            new_graph.is_interface = is_interface
            new_graph.neighbors_index = neighbors_index
            new_graph.neighbors_membership = neighbors_membership
            subdomain = self.create_subgraph(new_graph, id)
            new_membership = self.overlap_membership_one_subdomain(new_graph, subdomain)
            new_graph.membership = new_membership
            
        return new_membership

    def overlap_membership_one_subdomain(self, graph, subdomain):
        '''
        For one subdomain (NetworkX object), returns the new membership array 
        (w.r.t graph - NetworkX object) that corresponds to an overlapping of one. 
        '''
        #search where nodes of the subgraph are at the interface
        index_boundary = np.where(subdomain.is_interface == True)[0]
        #convert subgraph nodes to numpy array
        subgraph_nodes = np.asarray(list(subdomain.nodes()))
        #index (in the overall graph) of the nodes at the interface
        index_nodes_at_interface = subgraph_nodes[index_boundary]
        #index (in the overall graph) of the neighbors of each node at the interface
        index_neighbors_at_interface = [subdomain.neighbors_index[arg] for arg in index_boundary]
        #create new membership matrix for the subgraph
        membership_overlap = np.copy(graph.membership)
        #loop over the nodes at the interface
        for nb in range(len(index_nodes_at_interface)):
            #extract node index
            node = index_nodes_at_interface[nb]
            #extract neighbors index of the node
            node_neighbor = index_neighbors_at_interface[nb]
            #what is the partition of the considered node ?
            original_partition = graph.membership[node]
            #loop over the neighbor of the node
            for k in node_neighbor :
                #if membership of the neighbor is different from 
                #the membership of the considered node
                if membership_overlap[k] != original_partition :
                    #change it to be in the considered node's partition
                    membership_overlap[k] = original_partition
        
        return membership_overlap
    
    def dict_overlapping_cells(self, dict_subdomains):
        '''
        Returns a dictionnary of cells membership array of size (Ncells). 
        Associates each cell with its corresponding membership. If all vertices of
        the cell belongs to the same membership : value membership, else : 404.
        '''
        dict = {}
        for part in range(self.Nparts):
            #extract membership overlap subdomain and convert
            #in the order of mesh vertices (not dofs)
            membership_subdomain = dict_subdomains[str(part)]
            cell_membership = []
            #loop over the triangles
            for c in range(len(self.cells)):
                #extract triangles and each index
                triangle = self.cells[c]
                t1 = membership_subdomain[triangle[0]]
                t2 = membership_subdomain[triangle[1]]
                t3 = membership_subdomain[triangle[2]]
                if t1 == t2 and t1 == t3 and t2 == t3:
                    cell_membership.append(t1)
                else:
                    cell_membership.append(404)
            dict[str(part)] = np.asarray(cell_membership)

        return dict
    
    def check_membership(self, dict_subdomains):
        '''
        Returns a list of size Nnodes specifying in which subdomain belongs
        each node. Can be a list of multiple subdomains.'''
        
        full_list = []
        for nodes in range(self.full_graph.number_of_nodes()):
            sub_list = []
            for part in range(self.Nparts):
                sub_list.append((dict_subdomains[str(part)])[nodes])
            sub_list = list(set(sub_list))
            full_list.append(sub_list)

        return full_list 
    
    def dict_restriction_map(self, dict_subdomains):
        '''
        Returns a dictionnary with Nparts keys. For each key
        (i.e. subdomain) gives an array (size Nnodes of the subdomain)
        of restriction map. 
        Example : "0" : array([12,21,28,...]) means that for subdomain 0,
        the first node of this subdomain is the 21st node in the full graph.  
        '''
        dict = {}
        for part in range(self.Nparts):
            subdomain = dict_subdomains[str(part)]
            arg = np.where(subdomain == part)[0]
            dict[str(part)] = arg
        
        return dict

    def dict_boolean_interface(self, dict_subdomains):
        '''
        Returns a dictionnary with Nparts keys. For each key (i.e.)
        subdomain, gives a boolean array of size (Nnodes of the full graph),
        that indicates which nodes are at the interface.'''
        dict = {}
        for part in tqdm(range(self.Nparts)):
            subdomain = dict_subdomains[str(part)]
            is_interface, _, _ = self.interface(self.full_graph, subdomain)
            dict[str(part)] = is_interface
        
        return dict  

    def save_subdomains_to_xdmf(self, dict_subdomains, dict_cells):
        '''
        Convert all graph subdomains to mesh subdomains and save
        each partition independantly in XDMF / H5 format, using meshio.
        '''

        if os.path.exists(self.path_save_subdomains) :
            shutil.rmtree(self.path_save_subdomains)
            os.mkdir(self.path_save_subdomains)
        else : 
            os.mkdir(self.path_save_subdomains)

        for part in range(self.Nparts):
                
                # extract points of the subgraph
                argpoints = np.where(dict_subdomains[str(part)] == part)[0]
                points = self.full_graph.points[argpoints,:]

                # extract triangles of the subgraph
                argcells = np.where(dict_cells[str(part)] == part)[0]
                subdotriangles = self.cells[argcells,:]
                
                # convert all triangles from original mesh 
                # to extracted points indices
                for lign in range(np.shape(subdotriangles)[0]):
                    for column in range(np.shape(subdotriangles)[1]):
                        subdotriangles[lign][column] = np.argwhere(argpoints==subdotriangles[lign][column])[0][0]
                
                # create meshio Mesh object
                mesh_to_write = meshio.Mesh(points = points,
                                            cells = {'triangle' : subdotriangles}
                                            )
                
                # write the mesh in path
                mesh_to_write.write(os.path.join(self.path_save_subdomains, 'subdomain_{}.xdmf'.format(part)))