import numpy as np
from os.path import join


def built_adjacency_matrix(folder,mode='train'):
    """
        file: path to the file of adjacency list format like gowalla or amazon dataset
        mode : 'train' 'test' or 'val'
        return: NxN array adjacency matrix, with N the number of nodes
    
    """

    with open(join(folder,mode+'.txt')) as f : 
        lines_adj_list = [line.rstrip() for line in f]

    with open(join(folder,'item_list.txt')) as f : 
        lines_items = [line.rstrip() for line in f]

    N = len(lines_adj_list)
    M = len(lines_items)

    adj = np.zeros((N,M)) # bipartie graph

    for l in lines_adj_list:

        raw = [int(i) for i in l.split()] # first element is the user id
        user = raw[0]
        items = raw[1:]
        adj[user][items] = 1



    return adj

