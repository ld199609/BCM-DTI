
import numpy as np
from torch.utils import data
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
class NewDataset(data.Dataset):

    def __init__(self, smiles1, smiles2, label,words2idx_d,max_d):
        'Initialization'
        self.smiles1 = smiles1
        self.smiles2 = smiles2
        self.label = label
        self.words2idx_d=words2idx_d
        self.max_d=max_d
        # self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label)

    def drug2emb_encoder(self,x):
        max_d = self.max_d
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in x])  # index
        except:
            i1 = np.array([0])

        l = len(i1)

        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)

        else:
            i = i1[:max_d]

        return torch.FloatTensor(i)
    def data2tensor(self,data):

        return [torch.FloatTensor(d) for d in data]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        #d = self.df.iloc[index]['DrugBank ID']
        s1=self.smiles1[index]
        s2=self.smiles2[index]

        d_v = self.drug2emb_encoder(s1)
        p_v = self.drug2emb_encoder(s2)
        y = self.label[index]
        # print()
        return d_v, np.array([0]),p_v, y