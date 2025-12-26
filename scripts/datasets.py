# from ogb.utils import smiles2graph
import numpy as np
from ogb.utils.mol import smiles2graph
from ogb.lsc import PygPCQM4Mv2Dataset
from torch_geometric.loader import DataLoader
import torch
from tqdm import tqdm

def safe_smiles2graph(smiles):
    try:
        return smiles2graph(smiles)
    except AttributeError:
        # 如果遇到坏分子导致 RDKit 返回 None，就会触发这个错误
        # 我们返回一个只有一个孤立节点的“哑图”，保持数据对齐
        return {
            'edge_index': np.empty((2, 0), dtype=np.int64),
            'edge_feat': np.empty((0, 3), dtype=np.int64),
            'node_feat': np.zeros((1, 9), dtype=np.int64),
            'num_nodes': 1
        }

# PCQM4Mv2 数据集中包含了极少数（大约 370 万个中的几个）化学结构“不规范”的分子
# 新版的rdkit无法处理这些分子 修改 smiles2graph 函数

def get_ds(root, use_kfold=False, fold=None):
    pyg_dataset = PygPCQM4Mv2Dataset(root = root, smiles2graph = safe_smiles2graph)

    split_dict = pyg_dataset.get_idx_split()
    if use_kfold:
        assert fold is not None
        new_split_dict = torch.load(f'{root}/new_split_dict.pt')
        train_idx = new_split_dict[f'train_{fold}']
        valid_idx = new_split_dict[f'valid_{fold}']
    else:
        train_idx = split_dict['train']
        valid_idx = split_dict['valid']
    testdev_idx = split_dict['test-dev']
    testchallenge_idx = split_dict['test-challenge']

    train_dataset = pyg_dataset[train_idx]
    val_dataset = pyg_dataset[valid_idx]
    testdev_dataset = pyg_dataset[testdev_idx]
    testchallenge_dataset = pyg_dataset[testchallenge_idx]
    return train_dataset, val_dataset, testdev_dataset, testchallenge_dataset

def get_dl(root, config, use_kfold, fold=None, quick_run=False):

    train_dataset, val_dataset, testdev_dataset, testchallenge_dataset = get_ds(root, use_kfold, fold)
    batch_size = config.batch_size
    num_workers = 8

    if quick_run:
        train_dataset = train_dataset[:1000]
        val_dataset = val_dataset[:1000]
        testdev_dataset = testdev_dataset[:1000]
        testchallenge_dataset = testchallenge_dataset[:1000]

    train_dl = DataLoader(train_dataset, batch_size=batch_size, 
                        num_workers=num_workers, shuffle=True,
                        drop_last=True)
    valid_dl = DataLoader(val_dataset, batch_size=batch_size, 
                        num_workers=num_workers, shuffle=False,
                        drop_last=False)
    testdev_dl = DataLoader(testdev_dataset, batch_size=batch_size, 
                            num_workers=num_workers, shuffle=False,
                            drop_last=False)
    testchallenge_dl = DataLoader(testchallenge_dataset, batch_size=batch_size, 
                                num_workers=num_workers, shuffle=False,
                                drop_last=False)

    for batch in train_dl:
        break
    num_feas = batch.x.shape[1]
    print(f'num_feas: {num_feas}')

    return train_dl,valid_dl,testdev_dl,testchallenge_dl,num_feas

def gety(dl):
    ys = []
    for batch in dl:
        ys.append(batch.y)
    y = torch.cat(ys,dim=0)
    return y

def cal_mol_size(root):
    _, val_dataset, _, _ = get_ds(root, False)
    ns = []
    for i in tqdm(val_dataset,total=len(val_dataset)):
        ns.append(i.size()[0])
    torch.save(ns,f'{root}/valid_mol_size.pt')

    for fold in range(4):
        _, val_dataset, _, _ = get_ds(root, True, fold)
        ns = []
        for i in tqdm(val_dataset,total=len(val_dataset)):
            ns.append(i.size()[0])
        torch.save(ns,f'{root}/valid_mol_size_fold_{fold}.pt')

if __name__ == '__main__':
    from utils import load_yaml
    config = load_yaml('../yaml/gnn.yaml')
    from constants import PATH
    #get_dl(PATH, config, 0)
    #cal_mol_size(root=PATH)
    tr, val_dataset, _, _ = get_ds(PATH)
