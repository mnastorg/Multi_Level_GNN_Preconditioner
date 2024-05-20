##################### PACKAGES #################################################
################################################################################

import os
import sys

import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split
import json 
from json import JSONEncoder

import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data

################################################################################
################################################################################

class BuildDataset(InMemoryDataset):

    def __init__(self, root, transform = None, pre_transform = None, pre_filter = None, mode = None, precision = torch.float):
        
        self.mode = mode
        self.precision = precision
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        if self.mode == 'train' :
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.mode == 'val' :
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.mode == 'test' :
            self.data, self.slices = torch.load(self.processed_paths[2])
        else :
            sys.exit()

    @property
    def raw_file_names(self):
        files = ['A_sparse_matrix.npy', 
                 'b_matrix.npy', 
                 'sol.npy', 
                 'prb_data.npy', 
                 'tags.npy', 
                 'coordinates.npy']
        return files

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'data/')

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_dss/')

    def process(self):

        data_list = []

        list_A_sparse_matrix = np.load(self.raw_dir + self.raw_file_names[0], allow_pickle = True)
        list_b_matrix = np.load(self.raw_dir + self.raw_file_names[1], allow_pickle = True)
        list_sol = np.load(self.raw_dir + self.raw_file_names[2], allow_pickle = True)
        list_prb_data = np.load(self.raw_dir + self.raw_file_names[3], allow_pickle = True)
        list_tags = np.load(self.raw_dir + self.raw_file_names[4], allow_pickle = True)
        list_coordinates = np.load(self.raw_dir + self.raw_file_names[5], allow_pickle = True)

        for i in range(len(list_A_sparse_matrix)):
            
            # Build edge_index and a_ij
            A_sparse_matrix = list_A_sparse_matrix[i]
            coefficients = np.asarray(sc.sparse.find(A_sparse_matrix))
            edge_index = torch.tensor(coefficients[:2,:].astype('int'), dtype=torch.long)
            a_ij = torch.tensor(coefficients[2,:].reshape(-1,1), dtype=self.precision)

            # Build b tensor
            b =  torch.tensor(list_b_matrix[i].reshape(-1,1), dtype = self.precision)            

            # Extract exact solution
            sol = torch.tensor(list_sol[i].reshape(-1,1), dtype = self.precision)

            # Extract prb_data
            prb_data = torch.tensor(list_prb_data[i], dtype = self.precision)

            # Extract tags to differentiate nodes 
            tags = torch.tensor(list_tags[i], dtype=self.precision)
        
            # Extract coordinates
            pos = torch.tensor(list_coordinates[i], dtype = self.precision)
            
            data = Data(    edge_index = edge_index,
                            a_ij = a_ij, y = b, sol = sol,
                            prb_data = prb_data, tags = tags,  
                            pos = pos
                        )

            data_list.append(data)

        transform = T.Compose([T.Cartesian(), T.Distance()])
        data_list = [transform(data) for data in data_list]
            
        data_train_, data_test = train_test_split(data_list, test_size = 0.2, shuffle = True)
        data_train, data_val = train_test_split(data_train_, test_size = 0.25, shuffle = True)

        prb_data_stats = [data_train[i].prb_data for i in range(len(data_train))]
        prb_data_stats = np.vstack(prb_data_stats)
        prb_data_mean = np.mean(prb_data_stats, axis=0)
        prb_data_mean[np.abs(prb_data_mean) < 1.e-5] = 0.0
        prb_data_std = np.std(prb_data_stats, axis=0)
        prb_data_std[np.abs(prb_data_std) < 1.e-5] = 1.0
        
        dict_normalization = {  "prb_data_mean" : {"array" : prb_data_mean},
                                "prb_data_std" : {"array" : prb_data_std}
                                }

        # Serializing json
        json_object = json.dumps(dict_normalization, indent=4, cls=NumpyArrayEncoder)
        
        # Writing to sample.json
        with open(os.path.join(self.raw_dir, "normalization_info.json"), "w") as outfile:
            outfile.write(json_object)

        prb_data_mean = torch.tensor(prb_data_mean, dtype = self.precision)
        prb_data_std = torch.tensor(prb_data_std, dtype = self.precision)

        if self.mode == 'train' :

            for i in range(len(data_train)):
                data_train[i].prb_data = (data_train[i].prb_data - prb_data_mean) / prb_data_std 

            data, slices = self.collate(data_train)
            torch.save((data, slices), self.processed_paths[0])

        elif self.mode == 'val' :

            for i in range(len(data_val)):
                data_val[i].prb_data = (data_val[i].prb_data - prb_data_mean) / prb_data_std 

            data, slices = self.collate(data_val)
            torch.save((data, slices), self.processed_paths[1])

        elif self.mode == 'test':

            for i in range(len(data_test)):
                data_test[i].prb_data = (data_test[i].prb_data - prb_data_mean) / prb_data_std 

            data, slices = self.collate(data_test)
            torch.save((data, slices), self.processed_paths[2])

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
