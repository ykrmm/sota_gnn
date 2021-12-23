
import torch.nn as nn
from torch.nn.init import xavier_normal_,normal_
import torch

class BPR_MF(nn.Module):
    def __init__(self,N,args,device) -> None:
        super().__init__(BPR_MF,self)
        
        self.E = nn.Parameter(xavier_normal_(torch.rand((N,args.emb_size),requires_grad=True)))
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
        
        users = users.to(self.device)
        pos = pos.to(self.device)
        neg = neg.to(self.device)
        
        users_emb = self.E[users]
        pos_emb = self.E[pos]
        neg_emb = self.E[neg]
        
        pos_sim = (users_emb * pos_emb).sum(dim=-1) # Compute batch of similarity between users and positive items
        neg_sim = (users_emb * neg_emb).sum(dim=-1) # Compute batch of similarity between users and negative items
        
        return pos_sim,neg_sim
    
    
    def compute_score(self,n_users):
        """

        Args:
            n_users ([type]): Number of users
            
        return the similarity scores between each users and items of size N*M
        Use for test
        """
        users = self.E[:n_users] # N first lines of the embeddings matrix are for the users
        items = self.E[n_users:] # The others are for the items
        scores = users @ items.T # Inner product between all users and items
        return scores


