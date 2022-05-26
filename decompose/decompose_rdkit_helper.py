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

def load_rdkit_fragment(data,decompose_func):
    global frag
    s_dict={}
    p_dict={}
    s_list=[]
    p_list=[]
    l_list=[]
    smiles_list=[]
    for i,line in enumerate(data):
        print(i,len(data),dir)
        s,p,l=line.strip().split(" ")
        if p not  in p_dict.keys():
            type_pro=protein2category(p)
            p_frag=to_gram_no_overlap(type_pro)
            p_dict[p]=p_frag
        else:
            p_frag=p_dict[p]

        if s not in s_dict.keys():
            try:
                mol=Chem.MolFromSmiles(s)
                frag=decompose_func(mol)
                if frag :
                    s_dict[s] = frag
                    smiles_list.append(s)
                    s_list.append(frag)
                    p_list.append(p_frag)
                    l_list.append(l)
            except:
                print("decompose timeout or other error")
        else:
            frag=s_dict[s]
            smiles_list.append(s)
            s_list.append(frag)
            p_list.append(p_frag)
            l_list.append(int(l))

    frag_s=list(set([s for v in list(s_dict.values()) for s  in v]))
    frag_p=list(set([s for v in list(p_dict.values()) for s in v]))

    np.save(input_path+"smiles",smiles_list)
    np.save(input_path+"smiles_"+fragname,s_list)
    np.save(input_path+"protein_category_3_gram_"+fragname,p_list)
    np.save(input_path+"label_"+fragname,l_list)

    np.save(input_path+"smiles_"+fragname+'_set',frag_s)
    np.save(input_path+"protein_category_3_gram_"+fragname+'_set',frag_p)



if __name__ == '__main__':
    # load_dataset()
    # load_decompose_dataset()
    func_config={"recap":mol2Recap,"brics":mol2BRICS}
    converter=deepsmiles.Converter(rings=True,branches=True)
    fragname="brics"
    for dir in reversed(dataset_config["datasetDir"]):
        original_path=dir+"/original/"
        input_path=dir+"/input/"
        if "GPCR" not in dir:
            continue
        filter_dp_data=load_txt(input_path+"data_filter_dp")
        print(dir,len(filter_dp_data))
        load_rdkit_fragment(filter_dp_data,func_config[fragname])

