import torch
import torch.nn as nn 
import numpy as np 

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_,xavier_normal_
import torch.nn.functional as F
#Pytorch Geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree







class Light_GCN_Layer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



class LightGCN_pyg(nn.Module):
    def __init__(self,N,emb_size=64,layer=3) -> None:
        """NGCF Model with pytorch geometric framework

        Args:
            N ([int]): Number of nodes in bipartite graph
            emb_size (int, optional): Size of the embeddings. Defaults to 64.
            layer (int, optional): Number of GCN Layer in the model. Defaults to 3.
        """
        super().__init__()
        self.gc1 = Light_GCN_Layer(emb_size, emb_size)
        self.gc2 = Light_GCN_Layer(emb_size, emb_size)
        self.gc3 = Light_GCN_Layer(emb_size, emb_size)
        self.dropout = nn.Dropout(p=0.1)
        self.l_relu = nn.LeakyReLU()
        self.E = nn.Parameter(xavier_normal_(torch.rand((N,emb_size),requires_grad=True)))
    
    def forward(self,edge_index):
        
        
        self.E1 = self.l_relu(self.gc1(self.E,edge_index))
        self.E1 = F.normalize(self.E1, p=2, dim=1)
        self.E1 = self.dropout(self.E1)
        self.E2 = self.l_relu(self.gc2(self.E1,edge_index))
        self.E2 = F.normalize(self.E2, p=2, dim=1)
        self.E2 = self.dropout(self.E2)
        self.E3 = self.l_relu(self.gc3(self.E2,edge_index))
        self.E3 = F.normalize(self.E3, p=2, dim=1)
        
        EF = self.fusion()
        return EF



    def fusion(self):
        """
        Concatenate all the embeddings for the final representations
        """
        a0 = a1 = a2 = a3 = 1/ (self.layer+1) # Can be learned
        
        EF = (a0* self.E) + (a1* self.E1) + (a2 * self.E2) + (a3 * self.E3)
        
        return EF
