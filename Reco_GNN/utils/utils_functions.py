import numpy as np
from os.path import join
from scipy.linalg import sqrtm
import torch
from torch import Tensor
import argparse
def build_biparti_graph(folder,mode='train'):
    """
        file: path to the file of adjacency list format like gowalla or amazon dataset
        mode : 'train' 'test' or 'val'
        return: NxM array bi-parti adjacency matrix, with N the number of nodes, M the number of items
    
    """

    with open(join(folder,mode+'.txt')) as f : 
        lines_adj_list = [line.rstrip() for line in f]

    with open(join(folder,'item_list.txt')) as f : 
        lines_items = [line.rstrip() for line in f]

    N = len(lines_adj_list)
    M = len(lines_items)

    R = np.zeros((N,M)) # bipartie graph

    for l in lines_adj_list:

        raw = [int(i) for i in l.split()] # first element is the user id
        user = raw[0]
        items = raw[1:]
        R[user][items] = 1



    return R



def build_adjacency_matrix(R):
    
    """ Take the biparti graph R and return the adjacency Matrix A"""
    
    
    M = len(R[0])
    N = len(R)
    
    RT = R.T
    
    m1 = np.zeros((N,N))
    m2 = np.zeros((M,M))
    fr = np.concatenate((m1,R),axis=1)
    sr = np.concatenate((m2,RT),axis=1)
    A = np.concatenate((fr,sr),axis=0)
    

    return A  


def build_laplacien(A):
    """ Take the biparti graph R and return the adjacency Matrix A"""
    
    d  = np.sum(A,axis=0) # Degree in 1-D NP array
    D = np.diag(d)

    # Compute the square root inverse on D_t and A_t 
    D_t_inv = np.linalg.inv(sqrtm(D))
    
    L = D_t_inv @ A @ D_t_inv
    
    return L


def compute_laplacien(folder,mode='train',save=True):
    R = build_biparti_graph(folder,mode)
    A = build_adjacency_matrix(R)
    L = build_laplacien(A)
    L = torch.Tensor(L)
    if save:
        torch.save(L,'L.pt')
        print('saving successfully the laplacien matrix')
    return L



def count_parameters(model):
    """Count parameters of a model

    Args:
        model ([nn.Module]): The pytorch model

    Returns:
        [int]: Number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def structured_bipartite_negative_sampling(edge_index,n_users, num_nodes= None,contains_neg_self_loops=False):
    r"""
    Original function
    https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.structured_negative_sampling
    Modified to work with bipartite graph
    
    Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        n_users (int): Number of users in the bipartite graph
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    rand = torch.randint(num_nodes, (row.size(0), ), dtype=torch.long)
    neg_idx = row * num_nodes + rand
    
    users_idx = torch.arange(n_users)

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx,users_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)