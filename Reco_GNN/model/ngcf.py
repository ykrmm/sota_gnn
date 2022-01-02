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
    def __init__(self, in_channels, out_channels,aggr='add',bias=False):
        super().__init__(aggr=aggr)  # "Add" aggregation (Step 5).
        self.lin1 = torch.nn.Linear(in_channels, out_channels,bias=bias) # No bias in NGCF
        self.lin2 = torch.nn.Linear(in_channels, out_channels,bias=bias) # No bias in NGCF
    def forward(self, emb, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index_sl, _ = add_self_loops(edge_index, num_nodes=emb.size(0))

        # Step 2: Linearly transform node feature matrix.
        x1 = self.lin1(emb) # First weight matrix
        x2 = self.lin2(emb) # Second weight matrix

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, emb.size(0), dtype=emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm1 = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        row, col = edge_index_sl
        deg = degree(col, emb.size(0), dtype=emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm2 = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index_sl, emb=emb,x1=x1,x2=x2, norm1=norm1,norm2=norm2)

    def message(self, emb,x1,x2, norm1,norm2):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        print(emb.shape,x1.shape,x2.shape,norm1.shape,norm2.shape)
        e =( norm2.view(-1,1) * x1) + (norm1.view(-1,1)* emb * x2) 
        return e



    
    
    
class NGCF(nn.Module):
    def __init__(self,N,args,device='cpu') -> None:
        """NGCF Model with pytorch geometric framework

        Args:
            N ([int]): Number of nodes in bipartite graph ( Warning it's # users + items ! )
            args:
            - emb_size (int, optional): Size of the embeddings. Defaults to 64.
            - aggr (str,optional): Aggregation function in convolution layer. 'add' 'mean' or 'max'
            - pool (str,optional): Pooling mode for final representation. 'concat' 'mean' or 'sum'
            - negative_slope (float): negative slope for the leakyRelu
            - pretrain (bool): Use the pretrain embeddings matrix
            
        """
        super().__init__()
        emb_size = args.emb_size
        
        self.gc1 = NGCF_Layer(emb_size, emb_size,bias=False)
        self.gc2 = NGCF_Layer(emb_size, emb_size,bias=False) # ATTENTION on ajoute des self loops dans les couches de convolutions.
        self.gc3 = NGCF_Layer(emb_size, emb_size,bias=False)
        self.dropout = nn.Dropout(p=0.1,inplace=False)
        self.l_relu = nn.LeakyReLU(inplace=False,negative_slope=args.negative_slope)
        if args.pretrain:
            try:
                self.E = torch.load('embeddings/pretrain_emb.pt')
                print('sucess to load pretrain embeddings')
            except : 
                print('no pretrain embeddings found')
                raise NameError('Didnt find the pretrain embeddings')
                
        else:
            self.E = nn.Parameter(xavier_normal_(torch.rand((N,emb_size),requires_grad=True)))
        self.device = device
        self.pool = args.pool
    
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
        
        users = users.to(self.device)
        pos = pos.to(self.device)
        neg = neg.to(self.device)
        
        edge_index_direct = torch.stack((users,pos)) # Real connections in graph
        
        edge_index = to_undirected(edge_index_direct).to(self.device) # Propagate msg for users AND items 
        
        e1 = self.l_relu(self.gc1(self.E,edge_index))
        e1 = F.normalize(e1, p=2, dim=1)
        e1 = self.dropout(e1)
        e2 = self.l_relu(self.gc2(e1,edge_index))
        e2 = F.normalize(e2, p=2, dim=1)
        e2 = self.dropout(e2)
        e3 = self.l_relu(self.gc3(e2,edge_index))
        e3 = F.normalize(e3, p=2, dim=1) # Not sure if i have to normalize here
        
        self.ef = self._pooling((self.E,e1,e2,e3)) # Final representation is the concatenation of rpz of all layers
        
        users_emb = self.ef[users]
        pos_emb = self.ef[pos]
        neg_emb = self.ef[neg]
        
        pos_sim = (users_emb * pos_emb).sum(dim=-1) # Compute batch of similarity between users and positive items
        neg_sim = (users_emb * neg_emb).sum(dim=-1) # Compute batch of similarity between users and negative items
        
        return pos_sim,neg_sim
        
        
    def _pooling(self,emb_tuple):
        """

            Pooling of representations at each layers with mode
            mode (str): 'concat', 'sum' or 'mean'
        """
        if self.pool =='concat':
            ef = torch.cat(emb_tuple,1)
            
        if self.pool =='mean':
            ef = emb_tuple[0]
            for e in emb_tuple[1:]:
                ef = ef + e 
                
            return 1/len(emb_tuple) * ef # Simple mean, could add some attention factor 
            
        elif self.pool =='sum':
            ef = emb_tuple[0]
            for e in emb_tuple[1:]:
                ef = ef + e 
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
        