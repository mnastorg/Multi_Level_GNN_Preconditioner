######################################################################
############################# PACKAGES ###############################
######################################################################
import os 

import numpy as np 

import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing, MLP
import torch.autograd as autograd

from torch_geometric.utils import remove_self_loops

from utilities import utils 
utils.set_seed()
######################################################################
######################################################################
######################################################################

######################################################################
############################# MAIN MODEL #############################
######################################################################

class DeepStatisticalSolver(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        #Trainable blocks
        self.phi_to_list = nn.ModuleList([Phi_to([2*self.config["latent_dim"] + 3, 
                                                  self.config["latent_dim"], 
                                                  self.config["latent_dim"]], 
                                                  nn.ReLU()) 
                                                  for i in range(self.config["k"])])
        
        self.phi_from_list = nn.ModuleList([Phi_from([2*self.config["latent_dim"] + 3, 
                                                      self.config["latent_dim"], 
                                                      self.config["latent_dim"]], 
                                                      nn.ReLU()) 
                                                      for i in range(self.config["k"])])

        self.psi_list = nn.ModuleList([Psi([3*self.config["latent_dim"] + 1, 
                                            self.config["latent_dim"], 
                                            self.config["latent_dim"]], 
                                            nn.ReLU()) 
                                            for i in range(self.config["k"])])
        
        self.decoder_list = nn.ModuleList([Decoder([self.config["latent_dim"], 
                                                    self.config["latent_dim"], 
                                                    1], nn.ReLU()) 
                                                    for i in range(self.config["k"])])

        self.mse_loss = nn.MSELoss()

    def forward(self, batch):
        
        #Initialisation
        H, U = {}, {}

        cumul_res, cumul_mse, cumul_rel = {}, {}, {}
        loss_dic = {}
        total_loss = None

        sparse_matrix = SparseTensor(   row = batch.edge_index[0], 
                                        col = batch.edge_index[1], 
                                        value = batch.a_ij.ravel(), 
                                        sparse_sizes=(batch.num_nodes, batch.num_nodes)
                                    )

        H['0'] = torch.zeros([batch.num_nodes, self.config["latent_dim"]], dtype = torch.float, device = batch.sol.device)

        U['0'] = self.decoder_list[0](H['0'])

        cumul_res['0'] = self.residual_loss(U['0'], sparse_matrix, batch)

        cumul_mse['0'] = self.mse_loss(U['0'], batch.sol)

        cumul_rel['0'] = torch.linalg.norm(U['0'] - batch.sol)/torch.linalg.norm(batch.sol)

        for update in range(self.config["k"]) :

            mess_to = self.phi_to_list[update](H[str(update)], batch.edge_index, batch.edge_attr)
            mess_from = self.phi_from_list[update](H[str(update)], batch.edge_index, batch.edge_attr)

            concat = torch.cat([H[str(update)], mess_to, mess_from, batch.prb_data], dim = 1)

            correction = self.psi_list[update](concat)

            H[str(update+1)] = H[str(update)] + self.config["alpha"]*correction

            U[str(update+1)] = self.decoder_list[update](H[str(update+1)])

            cumul_res[str(update+1)] = self.residual_loss(U[str(update+1)], sparse_matrix, batch)

            cumul_mse[str(update+1)] = self.mse_loss(U[str(update+1)], batch.sol)

            cumul_rel[str(update+1)] = torch.linalg.norm(U[str(update+1)] - batch.sol) / torch.linalg.norm(batch.sol)

            if total_loss is None :
                total_loss = cumul_res[str(update+1)] * self.config["gamma"]**(self.config["k"] - update - 1)
            else :
                total_loss += cumul_res[str(update+1)] * self.config["gamma"]**(self.config["k"] - update - 1)
        
        loss_dic["train_loss"] = total_loss
        loss_dic["residual_loss"] = cumul_res
        loss_dic["mse_loss"] = cumul_mse
        loss_dic["rel_loss"] = cumul_rel

        return U, loss_dic

    def residual_loss(self, u, sparse_matrix, batch):
        
        residual =  sparse_matrix.matmul(u) - batch.y

        return torch.mean(residual**2)

######################################################################
######################################################################
######################################################################

######################################################################
############################# DL NETWORKS ############################
######################################################################

def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MLP(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super().__init__()

        layers = []
        units = hidden_channels[0]
        for k in range(1, len(hidden_channels)):
            next_units = hidden_channels[k]
            layers.append(nn.Linear(units, next_units))
            if k != len(hidden_channels) - 1 : layers.append(activation)
            units = next_units

        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):

        return self.mlp(x)

class Phi_to(MessagePassing):
    def __init__(self, hidden_channels=None, activation=None):
        super(Phi_to, self).__init__(aggr = 'add', flow = 'source_to_target')

        self.mlp = MLP(hidden_channels, activation)

    def forward(self, x, edge_index, edge_attr):
        
        edge_index, edge_attr = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)

        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)

        return self.mlp(tmp)

class Phi_from(MessagePassing):
    def __init__(self, hidden_channels=None, activation=None):
        super(Phi_from, self).__init__(aggr = 'add', flow = "target_to_source")

        self.mlp = MLP(hidden_channels, activation)

    def forward(self, x, edge_index, edge_attr):

        edge_index, edge_attr = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)

        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self, x_i, x_j, edge_attr):

        tmp = torch.cat([x_i, x_j, edge_attr], dim = 1)

        return self.mlp(tmp)

class Psi(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super(Psi, self).__init__()

        self.mlp = MLP(hidden_channels, activation)
    
    def forward(self, x):
        return self.mlp(x)

class Decoder(nn.Module):
    def __init__(self, hidden_channels=None, activation=None):
        super(Decoder, self).__init__()

        self.mlp = MLP(hidden_channels, activation)
    
    def forward(self, x):
        return self.mlp(x)


######################################################################
######################################################################
######################################################################
