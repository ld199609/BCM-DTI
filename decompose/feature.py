
from rdkit.Chem import FunctionalGroups
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit import Chem,RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem

from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem import Recap,BRICS
import scipy.sparse as sp

from utils.file_helper import *
import torch
import networkx as nx
fgs = FunctionalGroups.BuildFuncGroupHierarchy()
# one-hot编码，没有占位符
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

# 带有占位符的one-hot编码
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# def protein_to_PSSM(protein):
# rdkit.Chem.rdchem module：module containing the core chemistry functionality of the RDKit
def atom_features(atom, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    # 特征1：atom type:分子类型
    symbol = ['C', 'Co', 'P', 'K', 'Br', 'B', 'As', 'F', 'Ca', 'La', 'O', 'Au', 'Gd', 'Na', 'Se', 'N', 'Pt', 'S', 'Al',
              'Li', 'Cl', 'I', "other"]  # 23-dim
    # 特征2：degree of atom:分子的度    有多少化学键相连
    degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 11-dim
    # 特征3: hybridization type:杂化轨道类型
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    #   atom.GetSymbol()：             原子类型       23个维度
    #   atom.GetDegree()：             原子的度编码    11个维度                     原子的度
    #   atom.GetFormalCharge()：       原子的formal charge: 1个维度                形式电荷
    #   atom.GetNumRadicalElectrons()：原子的number of radical electrons: 1个维度  自由基的数目
    #   atom.GetHybridization()：      hybridization type: 6个维度                杂化轨道类型：
    #   atom.GetIsAromatic()：        1个维度,                                   是否是芳香族
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + [atom.GetIsAromatic()]  # 43

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  #
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    # return np.array(adjacency) + np.eye(adjacency.shape[0])
    return np.array(adjacency)  # do not use self-loop

def mol_features(mol):
    # -----------------01 将SMILE字符串转换为分子表示--------------
    num_atom_feat = 46
    # try:
    #     mol = Chem.MolFromSmiles(smiles)
    # except RuntimeError:
    #     print("SMILES cannot been parsed!")
    #     # raise RuntimeError("SMILES cannot been parsed!")
    # except:
    #     print("NEXT")
    # mol = Chem.AddHs(mol)
    # -----------------02 初始化分子图的节点特征矩阵--------------
    # 创建单个分子图的特征矩阵(分子数目, 分子的特征维度)
    try:
        # print(mol.GetNumAtoms())
        atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat),dtype=np.float64)
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = atom_features(atom)
        adj_matrix = adjacent_matrix(mol)
        # 05 对分子特征矩阵进行row_normalized 同时将邻接矩阵变为无向的邻接矩阵
        # 感觉进行normalized的意义不是太大
        # atom_normalized_feat = preprocess_features(atom_feat)
        # assert the graph is undirected graph
        adj1 = sp.csr_matrix(adj_matrix)
        undirected_adj_matrix = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
        # 06 返回分子的normalized节点特征以及对称的拓扑结构矩阵
        return True,atom_feat, undirected_adj_matrix.todense()
    except:
        print("AttributeError: NoneType object has no attribute GetNumAtoms")
    # -----------------03 遍历单个分子, 初始化分子特征------------
    # 遍历单个分子获取每个分子
    # for atom in mol.GetAtoms():
    #     atom_feat[atom.GetIdx(), :] = atom_features(atom)
    # no Z score
    # mean_value = np.mean(atom_feat, axis=0)
    # std_value = np.std(atom_feat, axis=0)
    # std_value[np.where(std_value == 0)] = 0.001
    # atom_feat = (atom_feat - mean_value) / std_value
    # -----------------04 获得分子的拓扑结构矩阵--------------
    return False,[],[]
def get_functional_group(mol):
    """
    extract functional group
    :param mol:
    :return:
    """
    function_group_list=[]
    for x in fgs:
        patt=x.pattern
        if mol.HasSubstructMatch(patt):
            function_group_list.append(Chem.MolToSmarts(patt))
    # if function_group_list:
    #     return function_group_list
    # else:
    #     return [Chem.MolToSmarts(mol)]
    return function_group_list
def mol_to_finger(mol):
    """
    extract chemical finger
    :param mol:
    :return:
    """
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3,trianglePruneBins=False)
    sigFactory.SetBins([(0,2),(2,5),(5,8)])
    sigFactory.Init()
    sigFactory.GetSigSize()
    fp = Generate.Gen2DFingerprint(mol,sigFactory)
    return [int(bit) for bit in list(fp.ToBitString()[0:50])]

def molecule_to_distance_matrix(mol):
    """
    extract distance matrix
    :param mol:
    :return:
    """
    # print("molecule: ",molecule_smiles)
    try:
        mol = Chem.AddHs(mol) # 加氢
        AllChem.EmbedMolecule(mol, randomSeed=1)      #通过距离几何算法计算3D坐标
        dm = AllChem.Get3DDistanceMatrix(mol)
        if dm.shape[0]==1 or np.isnan(dm).any() or np.all(dm==0):
            return False, []
        return True,dm
    except:
        return False,[]


def molecule_to_adjacency_matrix(mol):
    """
    extract adjacency matrix
    :param mol:
    :return:
    """
    try:
        adj_matrxi= Chem.GetAdjacencyMatrix(mol)
        if np.isnan(adj_matrxi).any()  or np.all(adj_matrxi==0) :
            # print("error")
            return False,[]
        # print("yes")
        return True ,adj_matrxi
    except:
        # print("error1")
        return False, []
def mol_to_nx(mol):
    """
    create molecule graph
    :param mol:
    :return:
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G
def smiles2atom(mol):
    re_atom=[]
    for atom in mol.GetAtoms():
        re_atom.append(atom.GetSymbol())
    return list(set(re_atom))

def get_3DDistanceMatrix(mol):
    """
    obtain distance matrix
    Args:
        trainFoldPath:

    Returns:

    """
    #
    # bm = molDG.GetMoleculeBoundsMatrix(mol)
    # # print(len(bm))
    # # mol2 = Chem.AddHs(mol) # 加氢
    # 生成3d构象
    AllChem.EmbedMolecule(mol, randomSeed=1)
    dm = AllChem.Get3DDistanceMatrix(mol)
    # dm_tensor = torch.FloatTensor([sl for sl in dm])
    return dm

def protein2smlies(p):
    p_mol = Chem.MolFromFASTA(p)
    p_smiles = Chem.MolToSmiles(p_mol)
    return p_mol,p_smiles

from func_timeout import func_set_timeout
@func_set_timeout(50)#设定函数超执行时间_
def mol2Recap(mol):
    hierarch = Recap.RecapDecompose(mol)
    res=list(hierarch.GetLeaves().keys())
    if res:
        return res
    return []
@func_set_timeout(50)#设定函数超执行时间_
def mol2BRICS(mol):
    # print(BRICS.BRICSDecompose(mol))
    try:
        res=list(BRICS.BRICSDecompose(mol))
        if res:
            return res
        return []
    except:
        return []

def load_subStructure_smiles():
    for root_path in reversed(DTI_dataset_config["datasetDir"]):
        # file_path=root_path+"/"+"original"
        out_path=root_path+"/"+"input"+"/"+"smiles"
        file_path=root_path+"/"+"input"+"/"+"smiles_sub"
        smiles=load_npy(file_path)
        recap_list=[]
        brics_list=[]
        for i,s in enumerate(smiles):
            mol=Chem.MolFromSmiles(s)
            recap=mol2Recap(mol)
            brics=mol2BRICS(mol)
            recap_list.append(recap)
            brics_list.append(brics)
        print(recap_list)
        np.save(out_path+"_recap",recap_list)
        np.save(out_path+"_brics",brics_list)
def illegal(s):
    mol=Chem.MolFromSmiles(s)
    if mol is not None:
        atom_num=len(mol.GetAtoms())
        if atom_num>=2:
            try:
            # bm = molDG.GetMoleculeBoundsMatrix(mol)
                mol = Chem.AddHs(mol) # 加氢
                AllChem.EmbedMolecule(mol, randomSeed=1)      #通过距离几何算法计算3D坐标
                dm = AllChem.Get3DDistanceMatrix(mol)
                dm_tensor = torch.FloatTensor([sl for sl in dm])
                return True
            except ValueError:
                # print("ValueError", s)
                return False
    return False
def dataset_filter():
    for root_path in reversed(DTI_dataset_config()["datasetDir"]):
        originalPath=root_path+"/"+"original"+"/"
        dataset=load_txt(originalPath+"data")
        f_filter=open(originalPath+"data_filter.txt","w")
        num=0
        for i,data in enumerate(dataset):
            s,p,l=data.split(" ")
            print(i,len(dataset),root_path)
            if '.' not in s and illegal(s):
                f_filter.write(" ".join([s,p,l]))
            else:
                num=num+1
                # data_filter.append([s,p,l])
        print("error: %d/%d"%(num,len(dataset)))
        f_filter.close()

@func_set_timeout(50)#设定函数超执行时间_
def to_gram_no_overlap(protein,k=3):
    data=protein
    i=0
    tmp=[]
    while i<len(data):
        if i+k<=len(data):
            tmp.append(data[i:i+k])
        else:
            tmp.append(data[i:-1])
        i=i+k
    return tmp


def protein2category(p):
    dict={}
    dict['H']='A'
    dict['R']='A'
    dict['K']='A'
    dict['D']='B'
    dict['E']='B'
    dict['N']='B'
    dict['Q']='B'
    dict['C']='C'

    dict['S']='D'
    dict['T']='D'
    dict['P']='D'
    dict['A']='D'
    dict['G']='D'

    dict['M']='E'
    dict['I']='E'
    dict['L']='E'
    dict['V']='E'

    dict['F']='F'
    dict['Y']='F'
    dict['W']='F'
    tmp=''
    for j,amino in enumerate(p.upper()):
        try:
            tmp=tmp+dict[amino]
        except:
            tmp=tmp+'G'
            print('bad amino',amino)
            # num=num+
    return tmp
if __name__ == '__main__':
    dataset_filter()
    # load_subStructure_smiles()