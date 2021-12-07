import numpy as np 
from os.path import join
from model import NGCF
from utils import compute_laplacien,Gowalla

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

data_path =  '/home/yannis/Documents/Recherche/Thèse/code/graph_datasets/gowalla'
data_path = '/Users/ykarmim/Documents/Recherche/Thèse/Code/graph_datasets/gowalla'




dataset = Gowalla(data_path,mode='train')

data = dataset.get_data()
