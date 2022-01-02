from os import write
import numpy as np 
from os.path import join
import torch
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from model import LightGCN,NGCF,BPR_MF
from utils import str2bool,Gowalla
from engine import train_one_epoch,eval_model
from time import time
import os
data_path =  '/home/yannis/Documents/Recherche/Thèse/code/graph_datasets/gowalla'
data_path = '/Users/ykarmim/Documents/Recherche/Thèse/Code/graph_datasets/gowalla'



def main():
    parser = ArgumentParser()
    # Model 
    parser.add_argument('--model', type=str, default='NGCF',help="Model is 'NGCF' 'LGCN' or 'MF' ")
    
    
    # Training
    parser.add_argument('--learning_rate', type=float, default=10e-4)
    parser.add_argument('--loss', type=str, default='BPR')
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--wd', type=float, default=2e-4)
    parser.add_argument('--moment', type=float, default=0.9)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--n_neg', default=1, type=int,help="Number of negative example")
    parser.add_argument('--shuffle', default=False, type=str2bool,help="Shuffle the edge index")
    parser.add_argument('--pretrain', default=False, type=str2bool,help="Use pretrain embeddings with MF")
    parser.add_argument('--negative_slope', type=float, default=0.01)
    # Model 
    
    parser.add_argument('--aggr',default='add',type=str,help="Type of aggregation function.'add','max' or 'mean' ")
    parser.add_argument('--pool', type=str, default='concat',help="Pooling of the representation at each layer. 'sum', 'mean' or 'concat' ")
    parser.add_argument('--emb_size',default=64,type=int,help='Size of the embeddings in the model')
    
    # Evaluation 
    parser.add_argument('--K', default=20, type=int,help="Evaluation on top K items")
    # Dataset and device
    parser.add_argument('--data_path',type=str,default='/Users/ykarmim/Documents/Recherche/Thèse/Code/graph_datasets/gowalla')
    parser.add_argument('--gpu', default=0, type=int,help="Wich gpu to select for training")
    
    # Saving 
    parser.add_argument('--log_dir',type=str,default=None,help="Log dir tensorboard")
    parser.add_argument('--save_path',type=str,default=None,help="If not none, model will be save in save_path")
    args = parser.parse_args()
    
    print('PARAMETERS')
    print(args)
    
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device used:",device)
    
    dataset_train = Gowalla(args.data_path,mode='train')
    dataset_test = Gowalla(args.data_path,mode='test')

    mask_train = dataset_train.get_mask()
    mask_test = dataset_test.get_mask()
    print('same number of nodes across mode?',dataset_train.data.num_nodes==dataset_test.data.num_nodes)
    
                    
    
    
    if args.model.upper() == 'NGCF':
        model = NGCF(dataset_train.data.num_nodes,args,device=device)
        
    elif args.model.upper() == 'LGCN':
        model = LightGCN(dataset_train.data.num_nodes,args,device=device)
        
        
    elif args.model.upper() == 'MF':
        model = BPR_MF(dataset_train.data.num_nodes,args,device=device)
        
    else: 
        raise ValueError("Model must be 'NGCF' or 'LGC'")

    model.to(device)
    
    # Optimizer 
    
    # Regularisation only on the embeddings matrix 
    decay = dict()
    no_decay = dict()
    for name, m in model.named_parameters():
        print('checking {}'.format(name))
        if 'E' in name:
            decay[name] = m 
        else:
            no_decay[name] = m 
    
    
    optimizer = torch.optim.Adam(
    [
        {"params": no_decay, "lr": args.learning_rate, "weight_decay": 0},
        {"params": decay, "lr": args.learning_rate, "weight_decay": args.wd},
    ],
    )
                    
            
            
    ndcg_max = -1
    recall_max = -1
    loss_min = 100
    writer = SummaryWriter(log_dir=args.log_dir)
    
    for ep in range(args.epoch):
        print('EPOCH',ep)
        train_s_t = time()
        
        if args.shuffle:
            dataset_train.shuffle()
        loss_train = train_one_epoch(model,optimizer,dataset_train,batch_size=args.batch_size)
        
        train_e_t = time()
        
        writer.add_scalar("Loss/train", loss_train, ep)
        
        
        if ep%5 == 0:
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "Loss train", "Recall train", "Ndcg train",\
                'Loss test', 'Recall Test', 'Ndcg test']
            
            _,ndcg_train,recall_train = eval_model(model,device,dataset_train,batch_size=args.batch_size,K=args.K,\
                mask=mask_test) # Eval trainset
            
            loss_test,ndcg_test,recall_test = eval_model(model,device,dataset_test,batch_size=args.batch_size,K=args.K,\
                mask=mask_train) # Eval testset
            
            recall_max = max(recall_max,recall_test)
            ndcg_max = max(ndcg_max,ndcg_test)
            loss_min = min(loss_min,loss_test)
            train_res.add_row(
                    [ep, train_e_t - train_s_t, loss_train, recall_train, ndcg_train, loss_test,
                     recall_test, ndcg_test])
            print(train_res)
            
            writer.add_scalar("Ndcg/train", ndcg_train, ep)
            writer.add_scalar("Recall/train", recall_train, ep)
            writer.add_scalar("Loss/test", loss_test, ep)
            writer.add_scalar("Ndcg/test", ndcg_test, ep)
            writer.add_scalar("Recall/test", recall_test, ep)
            
            
    # Keep track of the hyper_parameters     
    writer.add_hparams(
    {"lr": args.learning_rate, "batch_size": args.batch_size, "shuffle":args.shuffle,\
        "wd":args.wd,"pool":args.pool,"emb_size":args.emb_size,"pretrain":args.pretrain,\
            "negative_slope":args.negative_slope,"n_neg":args.n_neg,"model":args.model},
    {
        "Ndcg": ndcg_max,
        "Recall": recall_max,
        "Loss":loss_min
    },
    )
    writer.close()
    
    if args.model.upper() == 'MF':
        print('Saving pretrain embeddings...')
        torch.save(model.ef,'embeddings/pretrain_emb.pt')
        print('Embeddings matrix successfully save')
        
    if args.save_path is not None:
        torch.save(model,args.save_path)
if __name__ == '__main__':
    main()