# Source : https://github.com/guoyang9/BPR-pytorch


import numpy as np
import torch
from sklearn.metrics import ndcg_score



def dcg_at_k(r, k):
    
    return torch.sum(r / torch.log2(torch.arange(2, k + 2)))

	

def ndcg_metric(relevance,gt_rank, K):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is binary values. 
        Normalized discounted cumulative gain Binary case
    """
    relevance_gt = torch.zeros(K)
    relevance_gt[:len(gt_rank)] = 1
    idcg = dcg_at_k(relevance_gt,K)

    return dcg_at_k(relevance, K) / idcg


def recall_metric(relevance,gt_rank,K):
    
    
    recall = float(relevance.sum() / min(len(gt_rank),K))
    
    return recall





def compute_metrics(scores,K,dataset):
    top = torch.topk(scores,k=K)
    
    recall_list = []
    ndcg_list = []
    for u in range(len(scores)):
        recommand = top.indices[u]
        gt_rank = (dataset.R[u] == 1).nonzero().reshape(-1) # The ground truth rank is all the items that have a connection with our user. 
        relevance = (dataset.R[u][recommand]) # Get the true relevance of recommand items
        
        ndcg = ndcg_metric(relevance,gt_rank,K=K)
        recall = recall_metric(relevance,gt_rank,K=K)
        
        recall_list.append(recall)
        ndcg_list.append(ndcg)
        
    return np.array(ndcg_list).mean(),np.array(recall_list).mean()