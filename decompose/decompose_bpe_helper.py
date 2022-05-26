from utils.file_helper import *
import codecs
from subword_nmt.apply_bpe import BPE
from feature import *
def load_bpe_data(data):
    f_s=open(input_path+"smiles_"+fragname+".txt","w",encoding="utf-8")
    f_p=open(input_path+"protein_"+fragname+".txt","w",encoding="utf-8")
    f_catefory_p=open(input_path+"protein_category_"+fragname+".txt","w",encoding="utf-8")
    s_list=[]
    p_list=[]
    p_category_list=[]
    l_list=[]
    for line in data:
        s,p,l=line.split()
        p_category=protein2category(p)
        f_s.write(s+"\n")
        f_p.write(p+"\n")
        f_catefory_p.write(p_category+"\n")
        # print(s)
        s_list.append(s)
        p_list.append(p)
        p_category_list.append(p_category)
        l_list.append(l)
        # break
    f_s.close()
    f_p.close()
    f_catefory_p.close()
    execute_command("subword-nmt learn-bpe -s 1000 < "+input_path+"smiles_"+fragname+".txt> "+input_path+ "smiles_code.txt")
    execute_command("subword-nmt learn-bpe -s 2000 < "+input_path+"protein_"+fragname+".txt> "+input_path+ "protein_code.txt")
    execute_command("subword-nmt learn-bpe -s 2000 < "+input_path+"protein_category_"+fragname+".txt> "+input_path+ "protein_category_code.txt")

    print(len(s_list),len(p_list),len(l_list))
    np.save(input_path+"smiles",s_list)
    np.save(input_path+"protein",p_list)
    np.save(input_path+"protein_category_",p_category_list)

    np.save(input_path+"label",l_list)

def generate_bpe_dataset(filter_dp_data):
    smiles_codes = codecs.open(input_path +  "smiles_code.txt")
    protein_codes = codecs.open(input_path +  "protein_code.txt")
    protein_category_codes=codecs.open(input_path +  "protein_category_code.txt")
    smiles_code_bpe = BPE(smiles_codes, merges=-1, separator='')
    protein_code_bpe = BPE(protein_codes, merges=-1, separator='')
    protein_code_bpe= BPE(protein_category_codes, merges=-1, separator='')

    s_bpe_list = []
    p_bpe_list = []
    p_category_bpe_list=[]
    p_category_3_gram_list=[]

    label_list = []

    s_bpe_set=[]
    p_bpe_set=[]
    p_category_3_gram_set=[]
    p_category_bpe_set=[]

    for line in filter_dp_data:
        s, p, l = line.strip().split(" ")
        p_category = protein2category(p)
        p_category_3_gram = to_gram_no_overlap(p_category)

        s_bpe = smiles_code_bpe.process_line(s).split()
        p_bpe = protein_code_bpe.process_line(p).split()
        p_category_bpe=protein_code_bpe.process_line(p_category).split()



        s_bpe_set.extend([l for l in s_bpe])
        p_bpe_set.extend([l for l in p_bpe])
        p_category_bpe_set.extend([l for l in p_category_bpe])
        p_category_3_gram_set.extend([l for l in p_category_3_gram])

        s_bpe_list.append(s_bpe)
        p_bpe_list.append(p_bpe)
        p_category_bpe_list.append(p_category_bpe)
        p_category_3_gram_list.append(p_category_3_gram)
        label_list.append(l)

    np.save(input_path + "smiles_" + fragname, s_bpe_list)
    np.save(input_path + "protein_" + fragname, p_bpe_list)
    np.save(input_path+"protein_category_"+fragname,p_category_bpe_list)
    np.save(input_path+"protein_category_3_gram_"+fragname,p_category_3_gram_list)
    np.save(input_path + "label_" + fragname, label_list)

    np.save(input_path+"smiles_"+fragname+"_set",list(set(s_bpe_set)))
    np.save(input_path + "protein_" + fragname+"_set", list(set(p_bpe_set)))
    np.save(input_path+"protein_category_"+fragname+"_set",list(set(p_category_bpe_set)))
    np.save(input_path+"protein_category_3_gram_"+fragname+"_set",list(set(p_category_3_gram_set)))

def thesis_case(s,input_path):
    smiles_codes = codecs.open(input_path + "smiles_code.txt")
    smiles_code_bpe = BPE(smiles_codes, merges=-1, separator='')

    print( smiles_code_bpe.process_line(s).split())


def load_moltrans_bpe_data(data):
    fragname="bpe_moltrans"
    protein_smiles_dict=load_json(input_path+"protein2smiles")

    f_s = open(input_path + "smiles_" + fragname + ".txt", "w", encoding="utf-8")
    f_p = open(input_path + "protein_" + fragname + ".txt", "w", encoding="utf-8")
    f_catefory_p = open(input_path + "protein_category_" + fragname + ".txt", "w", encoding="utf-8")
    s_list = []
    p_list = []
    p_category_list = []
    l_list = []
    for line in data:
        s, p, l = line.split()
        # if p not in protein_smiles_dict.keys() or protein_smiles_dict[p]=="":
        #     continue
        # p=protein_smiles_dict[p]
        p_category = protein2category(p)
        f_s.write(s + "\n")
        f_p.write(p + "\n")
        f_catefory_p.write(p_category + "\n")
        # print(s)
        s_list.append(s)
        p_list.append(p)
        p_category_list.append(p_category)
        l_list.append(l)
        # break
    f_s.close()
    f_p.close()
    f_catefory_p.close()
    #
    # execute_command(
    #     "subword-nmt learn-bpe -s 500 < " + input_path + "smiles_" + fragname + ".txt> " + input_path + "smiles_code_500.txt")
    # execute_command(
    #     "subword-nmt learn-bpe -s 500 < " + input_path + "protein_" + fragname + ".txt> " + input_path + "protein_code_500.txt")
    # execute_command(
    #     "subword-nmt learn-bpe -s 500 < " + input_path + "protein_category_" + fragname + ".txt> " + input_path + "protein_category_code_500.txt")

    print(len(s_list), len(p_list), len(l_list))
    np.save(input_path + "smiles_"+fragname, s_list)
    np.save(input_path + "protein_"+fragname, p_list)
    np.save(input_path + "protein_category_"+fragname, p_category_list)

    np.save(input_path + "label_"+fragname, l_list)


if __name__ == '__main__':
    dataset_dir=DTI_dataset_config()
    fragname="bpe"
    for dir in reversed(dataset_dir["datasetDir"]):
        input_path="../"+dir+"/input/"
        if "celegans"   in dir or "human"  in dir or "DAVIS" in dir or "BindingDB" in dir:
            filter_dp_data=load_txt(input_path+"data_filter_dp")
            load_moltrans_bpe_data(filter_dp_data)
            continue
        # print(dir,len(filter_dp_data))
        # load_bpe_data(filter_dp_data)
        # generate_bpe_dataset(filter_dp_data)
        # thesis_case("O=C(C)OC1CCCC1C(=O)O",input_path)
