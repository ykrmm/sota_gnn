from shutil import Error

from torch._C import dtype
from .utils_functions import build_laplacien
from os.path import join

import numpy as np 
path_amazon = '/home/yannis/Documents/Recherche/Thèse/code/graph_datasets/amazon-book'
path_gowalla = '/home/yannis/Documents/Recherche/Thèse/code/graph_datasets/gowalla'


import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
class Gowalla:

    def __init__(self,process_dir,mode='train'):

        self.process_dir = process_dir # directory where the preprocess files are

        if mode != 'train' and mode != 'val' and mode!='test':
            raise ValueError('mode must be train val or test')
        self.mode = mode


    def load_file(self):
        with open(join(self.process_dir,self.mode+'.txt')) as f : 
            self.lines_adj_list = [line.rstrip() for line in f]

        with open(join(self.process_dir,'item_list.txt')) as f : 
            self.lines_items = [line.rstrip() for line in f]
        
        
    
    def build_biparti_graph(self):

        self.load_file()
        N = len(self.lines_adj_list)
        M = len(self.lines_items)

        R = np.zeros((N,M)) # bipartie graph

        for l in self.lines_adj_list:

            raw = [int(i) for i in l.split()] # first element is the user id
            user = raw[0]
            items = raw[1:]
            R[user][items] = 1


        
        return R

    def build_edge_index(self):

        self.load_file()
        N = len(self.lines_adj_list)
        M = len(self.lines_items)
        edge_u = []
        edge_v = []

        for l in self.lines_adj_list:
            raw = [int(i) for i in l.split()] # first element is the user id
            user = raw[0]
            items = raw[1:]
            items = [i + N for i in items] # else items and users will share same nodes ID

            edge_u.extend([user]*len(items))
            edge_v.extend(items)

        self.edge_index = torch.Tensor([edge_u,edge_v]).long()

    def biparti_to_edge_index(self,R):

        R = self.biparti_graph()

        edge_u = []
        edge_v = []

        for n in range(len(self.R)):
            for m in range(len(self.R[0])):
                if R[n][m]==1:
                    edge_u.append(n)
                    edge_v.append(m)

        

        
        edge_index = torch.Tensor([edge_u,edge_v]).long()

        return edge_index


    def get_edge_index(self):

        self.build_edge_index()

        return self.edge_index


    def get_data(self):

        self.build_edge_index
        self.edge_index = to_undirected(self.edge_index) # undirected graph
        data = Data(edge_index=self.edge_index)

        return data

