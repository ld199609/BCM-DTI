# class decomposer:
#     def __init__(self,method,):
#         self.method=method
#     def decompose(self):
#         if self.method=="bpe":
#
#     def bpe(self):
#     def brackets(self):
#     def mmps(self):
#     def ngrams(self):
#     def random_walk(self):
#     def subgraph(self):
#     def wordnijia(self):
#
#
#
#
#
#
# class dataset_generator():
#     def __init__(self,dataset,frag,p_type):
#         self.dataset=dataset
#         self.frag=frag
#         self.p_type=p_type
#     def load_origin_data(self,path):
#         smiles=load_npy(path+"smiles")
#         if self.p_type==1:
#             protein=load_npy(path+"protein")
#         elif self.p_type==2:
#             protein=load_npy(path+"protein_category")
#         else:
#             protein=load_npy(path+"protein_category_3_gram")
#         label=load_npy(path+"label")
#         return smiles,protein,label
#
#     def dataset_generate(self,path):
#         s,p,l=self.load_origin_data(path)
#
from func_timeout import func_set_timeout
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