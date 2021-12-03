import numpy as np
from os.path import join
from scipy.linalg import sqrtm
import torch
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

