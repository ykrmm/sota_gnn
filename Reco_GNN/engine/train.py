
import torch_geometric as pyg
import torch
from operator import itemgetter
import numpy as np

from .metric import ndcg,recall
from .utils_engine import batch

def train_one_epoch(model,optimizer,dataset,batch_size,lamb=0.01):
    
    model.train()
    optimizer.zero_grad()
    output = model(dataset.edge_index)
    sample = pyg.utils.structured_negative_sampling(dataset.direct_edge_index)
    ndcg_list = []
    recall_list = []
    loss_list = []
    
    for users,pos,neg in batch(sample, n=batch_size):
        
        (users * pos).sum(dim=-1)
        users_emb = itemgetter(*users)(output)
        users_emb = torch.stack(users_emb,dim=0)
        
        pos_emb = itemgetter(*pos)(output)
        pos_emb = torch.stack(pos_emb,dim=0)
        
        neg_emb = itemgetter(*neg)(output)
        neg_emb = torch.stack(neg_emb,dim=0)
        
        pos_sim = (users_emb * pos_emb).sum(dim=-1)
        neg_sim = (users_emb * neg_emb).sum(dim=-1)
        try:
            loss = - (pos_sim - neg_sim).sigmoid().log().mean() + lamb*(torch.norm(model.E,p=2) + torch.norm(model.gc1.lin.weight,p=2)\
            + torch.norm(model.gc2.lin.weight,p=2) + torch.norm(model.gc3.lin.weight,p=2)) # BPR Loss
            
        except:
            loss = - (pos_sim - neg_sim).sigmoid().log().mean() + lamb*(torch.norm(model.E,p=2)) # BPR Loss for LightGCN
        print(loss)
        loss_list.append(loss.item())
        loss.backward()
        
    
    return np.array(loss_list).mean(),np.array(ndcg_list).mean(),np.array(recall_list).mean()