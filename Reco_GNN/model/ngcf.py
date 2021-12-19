import torch
import torch.nn as nn 
import numpy as np 
from operator import itemgetter
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_,xavier_normal_
import torch.nn.functional as F
#Pytorch Geometric
from torch_geometric.nn import MessagePassing,GCNConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import to_undirected


############# PYG 





class NGCF_Layer(MessagePassing):
    def __init__(self, in_channels, out_channels,aggr='add'):
        super().__init__(aggr=aggr)  # "Add" aggregation (Step 5).
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
    def __init__(self,N,emb_size=64,aggr='add',layer=3) -> None:
        """NGCF Model with pytorch geometric framework

        Args:
            N ([int]): Number of nodes in bipartite graph
            emb_size (int, optional): Size of the embeddings. Defaults to 64.
            layer (int, optional): Number of GCN Layer in the model. Defaults to 3.
            aggr (str,optional): Aggregation function in convolution layer 'add' 'mean' or 'max'
        """
        super().__init__()
        self.gc1 = NGCF_Layer(emb_size, emb_size,aggr=aggr)
        self.gc2 = NGCF_Layer(emb_size, emb_size,aggr=aggr)
        self.gc3 = NGCF_Layer(emb_size, emb_size,aggr=aggr)
        self.dropout = nn.Dropout(p=0.1,inplace=False)
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
        self.EF = self.fusion()
        return self.EF


    def fusion(self):
        """
        Concatenate all the embeddings for the final representation
        """
        
        EF = torch.cat((self.E,self.E1,self.E2,self.E3),1)
        
        return EF
    
    
    def compute_score(self,n_users):
        """

        Args:
            n_users ([type]): Number of users
            
        return the similarity scores between each users and items of size N*M
        """
        users = self.EF[:n_users] # N first lines of the embeddings matrix are for the users
        items = self.EF[n_users:] # The others are for the items
        scores = users @ items.T # Inner product between all users and items
        return scores
    
    
    
    
class NGCF(nn.Module):
    def __init__(self,N,emb_size=64,aggr='add',device='cpu') -> None:
        """NGCF Model with pytorch geometric framework

        Args:
            N ([int]): Number of nodes in bipartite graph ( Warning it's # users + items ! )
            emb_size (int, optional): Size of the embeddings. Defaults to 64.
            layer (int, optional): Number of GCN Layer in the model. Defaults to 3.
            aggr (str,optional): Aggregation function in convolution layer 'add' 'mean' or 'max'
        """
        super().__init__()
        self.gc1 = GCNConv(emb_size, emb_size)
        self.gc2 = GCNConv(emb_size, emb_size)
        self.gc3 = GCNConv(emb_size, emb_size)
        self.dropout = nn.Dropout(p=0.1,inplace=False)
        self.l_relu = nn.LeakyReLU(inplace=False)
        
        self.E = nn.Parameter(xavier_normal_(torch.rand((N,emb_size),requires_grad=True)))
        self.device = device
    
    def forward(self,batch):
        """Generate embeddings for a batch of users and items.

        Args:
            batch (tuple(torch.LongTensor)): Sample batch of users, positive and negative items
            batch[0] --> users 
            batch[1] --> pos items 
            batch[2] --> neg items sampled 
            
            Careful the sample are directed graph from users to items, make it indirect by using 'to_undirect' function 
            from pyg. 
            
            Return (Positive similarity, negative similarity) 
            
        """
        users,pos,neg = batch
        
        users.to(self.device)
        pos.to(self.device)
        neg.to(self.device)
        
        edge_index_direct = torch.stack((users,pos)) # Real connections in graph
        
        edge_index = to_undirected(edge_index_direct) # Propagate msg for users AND items 
        
        self.e1 = self.l_relu(self.gc1(self.E,edge_index))
        self.e1 = F.normalize(self.e1, p=2, dim=1)
        self.e1 = self.dropout(self.e1)
        self.e2 = self.l_relu(self.gc2(self.e1,edge_index))
        self.e2 = F.normalize(self.e2, p=2, dim=1)
        self.e2 = self.dropout(self.e2)
        self.e3 = self.l_relu(self.gc3(self.e2,edge_index))
        self.e3 = F.normalize(self.e3, p=2, dim=1) # Not sure if i have to normalize here
        
        self.ef = self._fusion() # Final representation is the concatenation of rpz of all layers
        
        users_emb = itemgetter(*users)(self.ef) # Get the users embeddings according to index in the sample
        users_emb = torch.stack(users_emb,dim=0) 
        
        pos_emb = itemgetter(*pos)(self.ef)
        pos_emb = torch.stack(pos_emb,dim=0)
        
        neg_emb = itemgetter(*neg)(self.ef)
        neg_emb = torch.stack(neg_emb,dim=0)
        
        pos_sim = (users_emb * pos_emb).sum(dim=-1) # Compute batch of similarity between users and positive items
        neg_sim = (users_emb * neg_emb).sum(dim=-1) # Compute batch of similarity between users and negative items
        
        return pos_sim,neg_sim
        
        
    def _fusion(self):
        """
            Concatenation of representations at each layers
        """
        
        ef = torch.cat((self.E,self.e1,self.e2,self.e3),1)
        
        return ef
    
    def compute_score(self,n_users):
        """

        Args:
            n_users ([type]): Number of users
            
        return the similarity scores between each users and items of size N*M
        Use for test
        """
        users = self.ef[:n_users] # N first lines of the embeddings matrix are for the users
        items = self.ef[n_users:] # The others are for the items
        scores = users @ items.T # Inner product between all users and items
        return scores
        