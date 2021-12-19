
import torch_geometric as pyg
import torch
from operator import itemgetter
import numpy as np

from .metric import compute_metrics
from .utils_engine import batch

def train_one_epoch(model,optimizer,dataset,batch_size,lamb=0.01,K=20):
    """Compute one epoch of model training

    Args:
        model ([torch.nn.Module]): Pytorch model
        optimizer (torch.nn.optim): Optimizer for training
        dataset (Object): dataset train
        batch_size (int): 
        lamb (float, optional): Regularization term for the L2 loss. Defaults to 0.01.

    Returns:
        [tuple]: tuple containing the mean loss,ndcg and recall value. 
        
    """
    model.train()
    
    
    sample = pyg.utils.structured_negative_sampling(dataset.direct_edge_index)
    loss_list = []
    
    for b in batch(sample, n=batch_size):
        # meme si les samples sont triés, un utilisateur peut être dans plusieurs batchs, je ne sais pas 
        # à quel point c problématique. 
        optimizer.zero_grad()
        
        pos_sim,neg_sim = model(b)
        try:
            loss = - (pos_sim - neg_sim).sigmoid().log().mean() + lamb*(torch.norm(model.E.copy(),p=2) + \
                torch.norm(model.gc1.lin.weight.copy(),p=2)\
            + torch.norm(model.gc2.lin.weight.copy(),p=2) + torch.norm(model.gc3.lin.weight.copy(),p=2)) # BPR Loss mean over the batch
            
        except:
            loss = - (pos_sim - neg_sim).sigmoid().log().mean() + lamb*(torch.norm(model.E.copy(),p=2)) # BPR Loss for LightGCN
        #print(loss)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    
    return np.array(loss_list).mean()


def eval_model(model,dataset,batch_size,K=20,lamb=0.01,mask=None):
    """[summary]

    Args:
        model ([type]): Pytorch model
        dataset ([type]): Test dataset
        batch_size (int)
        lamb (float): Hyperparameter for the L2 regularization term
        K (int, optional): Evaluation on top K recommandation. Defaults to 20.
        mask_train (torch.bool, optional): Training items to mask during evaluation. Defaults to None.
    """
    # TO DO : Renvoyer metriques top20,40,60
    model.eval()
    n_users = len(dataset.users_id)
    with torch.no_grad():
        scores = model.compute_score(n_users)
        scores[mask] = -1000 # Give a negative score to items in train set, so we'll not recommand items in train set.
        
        ndcg,recall = compute_metrics(scores,K,dataset)       
        sample = pyg.utils.structured_negative_sampling(dataset.direct_edge_index)
        output = model.EF
        loss_list = []
        for users,pos,neg in batch(sample, n=batch_size):
        
            users_emb = itemgetter(*users)(output) # Get the users embeddings according to index in the sample
            users_emb = torch.stack(users_emb,dim=0) 
            
            pos_emb = itemgetter(*pos)(output)
            pos_emb = torch.stack(pos_emb,dim=0)
            
            neg_emb = itemgetter(*neg)(output)
            neg_emb = torch.stack(neg_emb,dim=0)
            
            pos_sim = (users_emb * pos_emb).sum(dim=-1) # Compute batch of similarity between users and positive items
            neg_sim = (users_emb * neg_emb).sum(dim=-1) # Compute batch of similarity between users and negative items
            try:
                loss = - (pos_sim - neg_sim).sigmoid().log().mean() + lamb*(torch.norm(model.E,p=2) + torch.norm(model.gc1.lin.weight,p=2)\
                + torch.norm(model.gc2.lin.weight,p=2) + torch.norm(model.gc3.lin.weight,p=2)) # BPR Loss
                
            except:
                loss = - (pos_sim - neg_sim).sigmoid().log().mean() + lamb*(torch.norm(model.E,p=2)) # BPR Loss for LightGCN
            #print(loss)
            loss_list.append(loss.item())
    
    return np.array(loss_list).mean(),ndcg,recall