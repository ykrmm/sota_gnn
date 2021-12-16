import numpy as np 
from os.path import join
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

from model import NGCF_pyg,LightGCN_pyg
from utils import compute_laplacien,str2bool,Gowalla
from engine import train_one_epoch,eval_model
data_path =  '/home/yannis/Documents/Recherche/Thèse/code/graph_datasets/gowalla'
data_path = '/Users/ykarmim/Documents/Recherche/Thèse/Code/graph_datasets/gowalla'



def main():
    parser = ArgumentParser()
    # Model 
    parser.add_argument('--model', type=str, default='NGCF')
    
    
    # Training
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--loss', type=str, default='BPR')
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--lamb', type=float, default=0.01,help="Lambda parameter for the L2 Regularization")
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    
    # Model 
    
    parser.add_argument('--aggr',default='add',type=str,help="Type of aggregation function.'add','max' or 'mean' ")
    parser.add_argument('--emb_size',default=64,type=int,help='Size of the embeddings in the model')
    
    # Evaluation 
    parser.add_argument('--K', default=20, type=int,help="Evaluation on top K items")
    # Dataset and device
    parser.add_argument('--data_path',type=str,default='/Users/ykarmim/Documents/Recherche/Thèse/Code/graph_datasets/gowalla')
    parser.add_argument('--gpu', default=0, type=int,help="Wich gpu to select for training")
    args = parser.parse_args()
    
    
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    
    dataset_train = Gowalla(data_path,mode='train')
    dataset_test = Gowalla(data_path,mode='test')

    mask_train = dataset_train.get_mask()
    mask_test = dataset_test.get_mask()
    print('same number of nodes across mode?',dataset_train.data.num_nodes==dataset_test.data.num_nodes)
    writer = SummaryWriter()
    if args.model == 'NGCF':
        model = NGCF_pyg(dataset_train.data.num_nodes,emb_size=args.emb_size,aggr=args.aggr)
        
    elif args.model.upper() == 'LGCN':
        model = LightGCN_pyg(dataset_train.data.num_nodes,emb_size=args.emb_size,aggr=args.aggr)
        
    else: 
        raise ValueError("Model must be 'NGCF' or 'LGC'")

    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.wd)
    for ep in range(args.epoch):
        loss_train,ndcg,recall = train_one_epoch(model,optimizer,dataset_train,batch_size=args.batch_size,\
            lamb=args.lamb,K=args.K,mask_test= mask_test)
        
        loss_test,ndcg_test,recall_test = eval_model(model,dataset_test,batch_size=args.batch_size,K=args.K,\
            mask_train=mask_train)
        
        
if __name__ == '__main__':
    main()