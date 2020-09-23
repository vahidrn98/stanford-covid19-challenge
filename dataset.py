
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

target_col = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

rna_dict    = {x:i for i, x in enumerate('ACGU')} #4
struct_dict = {x:i for i, x in enumerate('().')}  #3
loop_dict   = {x:i for i, x in enumerate('BEHIMSX')}#7

class RNADataset(Dataset):
    def __init__(self, df, augment=None):

        self.rna    = df['sequence'].map(lambda seq: [rna_dict[x] for x in seq])
        self.struct = df['structure'].map(lambda seq: [struct_dict[x] for x in seq])
        self.loop   = df['predicted_loop_type'].map(lambda seq: [loop_dict[x] for x in seq])

        bbp0 =[]
        bbp1 =[]
        id = df.id.values
        for i in id:
            probability = np.load(f'../bpps/{i}.npy')
            bbp0.append(probability.max(-1).tolist())
            bbp1.append((1-probability.sum(-1)).tolist())
        self.bbp0 = bbp0
        self.bbp1 = bbp1

        #---
        if 'reactivity' in df.columns:
            target = np.transpose(
                df[target_col]
                .values
                .tolist(),
            (0, 2, 1))
            target = np.ascontiguousarray(target)
        else:
            target = np.zeros((len(df),1,1)) #dummy

        self.df =  df
        self.len = len(self.df)
        self.augment = augment
        self.target = target

    def __str__(self):
        string  = ''
        string += '\tlen  = %d\n'%len(self)
        return string

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        r = self.df.loc[index]
        target = self.target[index]

        rna = np.array(self.rna[index])
        struct = np.array(self.struct[index])
        loop = np.array(self.loop[index])
        bbp0 = np.array(self.bbp0[index]).reshape(-1,1)
        bbp1 = np.array(self.bbp1[index]).reshape(-1,1)

        #bbp = np.load(f'../input/stanford-covid-vaccine/bpps/{r.id}.npy')
        #bbp = np.expand_dims(bbp, axis=0)

        seq = np.concatenate([
            np_onehot(rna,4),
            np_onehot(struct,3),
            np_onehot(loop,7),
            bbp0,
            bbp1,
        ],1)

        #------
        record = {
            'target': torch.tensor(target, dtype=torch.float),
            #'bbps'  : torch.tensor(bbp, dtype=torch.float),
            'seq' : torch.tensor(seq, dtype=torch.float),
            'ids' : r.id
        }
        if self.augment is not None: record = self.augment(record)
        return record

def np_onehot(x, max=54):
    return np.eye(max)[x]
