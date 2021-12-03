import numpy as np 
from os.path import join
from model import NGCF
from utils import compute_laplacien



data_path =  '/home/yannis/Documents/Recherche/Thèse/code/graph_datasets/gowalla'
data_path = '/Users/ykarmim/Documents/Recherche/Thèse/Code/graph_datasets/gowalla'





L = compute_laplacien(data_path,mode='train')

ngcf = NGCF(len(L),64)
