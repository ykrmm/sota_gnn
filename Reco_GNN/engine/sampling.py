from torch import Tensor 
import torch 
import numpy as np 

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

    rand = torch.randint(low=n_users-1,high=num_nodes, size=(row.size(0), ),dtype=torch.long)
    neg_idx = row * num_nodes + rand
    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)