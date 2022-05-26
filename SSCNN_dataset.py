
import numpy as np
from torch.utils import data
import torch
class NewDataset(data.Dataset):

    def __init__(self, smiles, protein, label,words2idx_d,words2idx_p,max_d,max_p):
        'Initialization'
        self.smiles = smiles
        self.protein = protein
        self.label = label
        self.words2idx_d=words2idx_d
        self.words2idx_p=words2idx_p
        self.max_d=max_d
        self.max_p=max_p
        # self.transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.label)

    def protein2emb_encoder(self,x):
        max_p = self.max_p
        # i1=np.zeros(max_p)
        # for ch in x:
        #     index1=self.words2idx_p[ch]
        #     i1[index1]=1
        try:
            i1 = np.asarray([self.words2idx_p[i] for i in x])  # index
        except:
            i1 = np.array([0])
            #print(x)
        l = len(i1)

        if l < max_p:
            i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        else:
            i = i1[:max_p]

        return torch.FloatTensor(i)

    def drug2emb_encoder(self,x):
        max_d = self.max_d
        # i1=np.zeros(max_d)
        # for ch in x:
        #     index1=self.words2idx_d[ch]
        #     i1[index1]=1
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
        s=self.smiles[index]
        p=self.protein[index]

        # mol=Chem.MolFromSmiles(s)
        # finger = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=50))

        d_v = self.drug2emb_encoder(s)
        p_v = self.protein2emb_encoder(p)
        y = self.label[index]
        # print()
        return d_v, np.array([0]),p_v, y