# Load torch library 
import torch
import torch.nn as nn
import torch.nn.functional as F

class Two_Layer_GCN(nn.Module):
    def __init__(self,n_nodes,dim_x,dim_h,dim_out):
        # A_hat --> normalize adjacency matrix
        # X --> Matrix of features
        # dim_in --> Input dimension of X, A_hat 
        # dim_h --> Hidden dimension
        # dim_out --> Output dimension, number of classes 
        
        super(Two_Layer_GCN,self).__init__()
            
        # Layers
        self.W0 = nn.Parameter(torch.rand((dim_x,dim_h),requires_grad=True)) # Tell pytorch its a trainable parameter
        self.W1 = nn.Parameter(torch.rand((dim_h,dim_out),requires_grad=True)) 
        self.relu = nn.ReLU()
        #self.myparameters = nn.ParameterList(self.W0, self.W1)
        
    def embbed(self,A_hat,X):
        # Get the embeddings of the nodes (before softmax)
        A_hat = torch.Tensor(A_hat) # Dim N*N (N Number of nodes)
        X = torch.Tensor(X)  # N* dim_in 
        hidden = self.relu(A_hat @ X @ self.W0) # Dim N*H
        out = A_hat @ hidden @ self.W1 # Dim N*2
        return out
        
        
    def forward(self,A_hat,X):
        # Perform a forward pass
        A_hat = torch.Tensor(A_hat) # Dim N*N (N Number of nodes)
        X = torch.Tensor(X)  # N* dim_x 
        hidden = A_hat @ X @ self.W0 # Dim N*H
        hidden = self.relu(hidden)
        out = A_hat @ hidden @ self.W1# Dim N*2
        #out = self.relu(out)
        
        return out