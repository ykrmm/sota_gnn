import torch
import torch.nn as nn 
import numpy as np 

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_,xavier_normal_

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









