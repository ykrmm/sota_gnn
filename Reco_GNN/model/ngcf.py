import torch
import torch.nn as nn 
import numpy as np 

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_,xavier_normal_

#Pytorch Geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class NGCFConv(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, dim_in, dim_out, bias=True):
        super(NGCFConv, self).__init__()
        
        self.W0 = nn.Parameter(xavier_normal_(torch.rand((dim_in,dim_out),requires_grad=True))) # Tell pytorch its a trainable parameter
        self.W1 = nn.Parameter(xavier_normal_(torch.rand((dim_in,dim_out),requires_grad=True))) # Initialize with xavier_uniform like the NGCF paper
        if bias:
            self.bias = Parameter(torch.FloatTensor(dim_out))
        
        
    def forward(self,E,L):
        I =  torch.eye(len(L))
        
        E = ((L+I) @ E @ self.W1) + (L@E)*  ( E @ self.W2)
        if self.bias is not None:
            return E + self.bias
        else:
            return E

        
        
class NGCF(nn.Module):
    def __init__(self, dim_in, emb_size):
        super(NGCF, self).__init__()

        self.gc1 = NGCFConv(emb_size, emb_size)
        self.gc2 = NGCFConv(emb_size, emb_size)
        self.gc3 = NGCFConv(emb_size, emb_size)
        self.dropout = nn.Dropout(p=0.1)
        self.l_relu = nn.LeakyReLU()
        self.E = nn.Embedding(dim_in,emb_size)
    
    def forward(self, L):
        
        
        self.E = self.l_relu(self.gc1(self.E,L))
        self.E = self.dropout(self.E)
        self.E = self.l_relu(self.gc2(self.E,L))
        self.E = self.dropout(self.E)
        self.E = self.l_relu(self.gc3(self.E,L))
        
        return self.E



############# PYG 





class NGCF_Layer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



class NGCF_pyg(nn.Module):
    def __init__(self,N,emb_size=64,layer=3) -> None:
        """NGCF Model with pytorch geometric framework

        Args:
            N ([int]): Number of nodes in bipartite graph
            emb_size (int, optional): Size of the embeddings. Defaults to 64.
            layer (int, optional): Number of GCN Layer in the model. Defaults to 3.
        """
        super().__init__()
        self.gc1 = NGCF_Layer(emb_size, emb_size)
        self.gc2 = NGCF_Layer(emb_size, emb_size)
        self.gc3 = NGCF_Layer(emb_size, emb_size)
        self.dropout = nn.Dropout(p=0.1)
        self.l_relu = nn.LeakyReLU()
        #self.E = nn.Embedding(N,emb_size)
        
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
        Concatenate all the embeddings for the final representation
        """
        
        EF = torch.cat((self.E,self.E1,self.E2,self.E3),1)
        
        return EF