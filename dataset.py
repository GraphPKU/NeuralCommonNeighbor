import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, to_undirected

# random split dataset
def randomsplit(dataset, val_ratio: float=0.10, test_ratio: float=0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)
        ei = ei[:, ei[0]<ei[1]]
        return ei
    data = dataset[0]
    hasea = data.edge_attr is not None
    data.num_nodes = data.x.shape[0] if data.x is not None else torch.max(data.edge_index).item() + 1
    data = train_test_split_edges(data, test_ratio, test_ratio)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    num_val = int(data.val_pos_edge_index.shape[1] * val_ratio/test_ratio)
    data.val_pos_edge_index = data.val_pos_edge_index[:, torch.randperm(data.val_pos_edge_index.shape[1])]
    if hasea:
        split_edge['train']['edge'] = torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1).t()
        split_edge['train']['edge_attr'] = torch.cat((data.train_pos_edge_attr, data.val_pos_edge_attr[:-num_val]), dim=-1)
        split_edge['valid']['edge'] = data.val_pos_edge_index[:, -num_val:].t()
        split_edge['valid']['edge_attr'] = data.val_pos_edge_attr[-num_val:]
        split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
        split_edge['test']['edge'] = data.test_pos_edge_index.t()
        split_edge['test']['edge_attr'] = data.test_pos_edge_attr[-num_val:]
        split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    else:
        split_edge['train']['edge'] = removerepeated(torch.cat((data.train_pos_edge_index, data.val_pos_edge_index[:, :-num_val]), dim=-1)).t()
        split_edge['valid']['edge'] = removerepeated(data.val_pos_edge_index[:, -num_val:]).t()
        split_edge['valid']['edge_neg'] = removerepeated(data.val_neg_edge_index).t()
        split_edge['test']['edge'] = removerepeated(data.test_pos_edge_index).t()
        split_edge['test']['edge_neg'] = removerepeated(data.test_neg_edge_index).t()
    return split_edge

def loaddataset(name: str, use_valedges_as_input: bool, load=None):
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]
        data.edge_weight = None 
    elif name in ["Sample"]:
        import pickle
        from torch_geometric.utils import from_networkx
        with open('sample_data_nx_graph.pkl', 'rb') as f:
            G = pickle.load(f)
        pyg_graph = from_networkx(G)
        pyg_graph.edge_attr = pyg_graph.weight
        dataset = [pyg_graph]
        split_edge = randomsplit(dataset)
        data = dataset[0]
        data.edge_index, data.edge_attr = to_undirected(split_edge["train"]["edge"].t(), split_edge["train"]["edge_attr"])
        edge_index = data.edge_index

    elif name in ["collab", "ppa", "ddi", "citation2"]:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        edge_index = data.edge_index
        data.edge_weight = None 
    else:
        raise NotImplementedError
    # print(data.num_nodes, edge_index.max())
    data.adj_t = SparseTensor.from_edge_index(edge_index, edge_attr=data.edge_attr, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi" or "Sample":
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    print("dataset split ")
    for key1 in split_edge:
        for key2  in split_edge[key1]:
            print(key1, key2, split_edge[key1][key2].shape[0])


    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge

if __name__ == "__main__":
    loaddataset("Cora", False)
    loaddataset("Citeseer", False)
    loaddataset("Pubmed", False)
    loaddataset("ppa", False)
    loaddataset("collab", False)
    loaddataset("citation2", False)