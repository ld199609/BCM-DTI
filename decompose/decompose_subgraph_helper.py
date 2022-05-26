from utils.utils_helper import *
from dataset_config import *
import deepsmiles



doc2vec_config=doc2vec_config()
dataset_config=DTI_dataset_config()
word2vec_config=word2vec_config()

def filter_dp(input_path,data):
    f=open(input_path+"data_filter_dp.txt","w")
    for line in data:
        s,p,l=line.strip().split(" ")
        if '.' not in s and illegal(s):
            s_en=converter.encode(s)
            s_de=converter.decode(s_en)
            f.write(" ".join([s_de,p,l])+"\n")
        # break
    f.close()
def getMaxLengthMol(per_frag_list):
    maxLength=0
    for frag in per_frag_list:
        mol = Chem.MolFromSmiles(frag)
        if maxLength<mol.GetNumAtoms():
            maxLength=mol.GetNumAtoms()
    return maxLength

def load_subgraph(smiles_frag_list,protein_frag_list,label_frag_list):
    feature_all_list=[]
    adj_all_list=[]
    protein_to_write=[]
    label_to_write=[]
    protein_to_write_set=[]
    for i,per_frag_list in enumerate(smiles_frag_list):
        feature_list=[]
        adj_list=[]
        f_s=True
        length=getMaxLengthMol(per_frag_list)
        for frag in per_frag_list:
            mol=Chem.MolFromSmiles(frag)
            f,feat,adj=mol_features(mol)

            if f:
                feature_list.append(feat.tolist())
                adj_list.append(adj.getA().tolist())
            else:
                f_s=False
                break
        if f_s:
            print(dir,i,len(smiles_frag_list))
            feature_all_list.append(feature_list)
            adj_all_list.append(adj_list)
            protein_to_write.append(protein_frag_list[i])
            label_to_write.append(label_frag_list[i])
            protein_to_write_set.extend(protein_frag_list[i])

    np.save(input_path+"smiles_feat_"+to_write_fragname,feature_all_list)
    np.save(input_path+"smiles_adj_"+to_write_fragname,adj_all_list)
    np.save(input_path+"protein_category_3_gram_"+to_write_fragname,protein_to_write)
    np.save(input_path+"label_"+to_write_fragname,label_to_write)
    np.save(input_path + "protein_category_3_gram_" + to_write_fragname + "_set", list(set(protein_to_write_set)))
def load_gcnconv_dataset(smiles_frag_list,protein_frag_list,label_frag_list):
    feature_all_list=[] # the node feature of graph
    edge_all_list=[]# 2* the num of the edge
    protein_to_write=[]
    label_to_write=[]
    protein_to_write_set=[]
    for i,per_frag_list in enumerate(smiles_frag_list):
        feature_list=[]
        edge_list=[]
        f_s=True
        for frag in per_frag_list:
            mol=Chem.MolFromSmiles(frag)
            f,feat,adj=mol_features(mol)
            # print(feat)
            # print(adj)
            adj_tri_edge=list(np.nonzero(np.triu(adj,1)))
            if len(adj_tri_edge[0])==1:
                print(frag)
            # print(adj_tri_edge)
            if f and len(feat)>1 and len(adj_tri_edge[0])>1:
                # print(np.nonzero(adj_tri))
                edge_list.append(adj_tri_edge)
                feature_list.append(feat.tolist())
            else:
                f_s=False
                break
        if f_s:
            print(dir,i,len(smiles_frag_list))
            feature_all_list.append(feature_list)
            edge_all_list.append(edge_list)
            protein_to_write.append(protein_frag_list[i])
            label_to_write.append(label_frag_list[i])
            protein_to_write_set.extend(protein_frag_list[i])

    np.save(input_path+"smiles_feat_"+to_write_fragname,feature_all_list)
    np.save(input_path+"smiles_edge_"+to_write_fragname,edge_all_list)
    np.save(input_path+"protein_category_3_gram_"+to_write_fragname,protein_to_write)
    np.save(input_path+"label_"+to_write_fragname,label_to_write)
    np.save(input_path + "protein_category_3_gram_" + to_write_fragname + "_set", list(set(protein_to_write_set)))

if __name__ == '__main__':
    # load_dataset()
    # load_decompose_dataset()
    func_config={"recap":mol2Recap,"brics":mol2BRICS}
    converter=deepsmiles.Converter(rings=True,branches=True)
    fragname="brics"
    to_write_fragname="brics_subgraph"
    for dir in reversed(dataset_config["datasetDir"]):
        if "DUDE" in dir:
            continue
        original_path=dir+"/original/"
        input_path=dir+"/input/"
        filter_dp_data=load_txt(input_path+"data_filter_dp")
        print(dir,len(filter_dp_data))
        smiles_frag_list=load_npy(input_path+"smiles_"+fragname)
        protein_frag_list=load_npy(input_path+"protein_category_3_gram_"+fragname)
        label_frag_list=load_npy(input_path+"label_"+fragname)
        protein_frag_set=load_npy(input_path+"protein_category_3_gram_"+fragname+"_set")
        # load_subgraph(smiles_frag_list,protein_frag_list,label_frag_list)
        load_gcnconv_dataset(smiles_frag_list,protein_frag_list,label_frag_list)

        # np.save(input_path+"protein_category_3_gram_"+to_write_fragname,protein_frag_list)
        # np.save(input_path+"label_"+to_write_fragname,label_frag_list)
        # np.save(input_path+"protein_category_3_gram_"+to_write_fragname+"_set",protein_frag_set)


