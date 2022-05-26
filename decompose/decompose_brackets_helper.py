
from decompose.feature import *
# from decompose.decompose import *
from utils.file_helper import *
def extract_sub(s):

    stack_b=[]
    stack_sub=[]
    sub=[]
    tmp=''
    for ch in s:
        if ch =="(" or ch=='（':
            stack_sub.append(tmp)
            stack_b.append(ch)
            tmp=""

        elif ch==")"  or ch=="）":
            stack_b.pop()
            sub.append(tmp)
            tmp=stack_sub.pop()
        else:
            tmp=tmp+ch
    if tmp:
        sub.append(tmp)
    return sub
def decompose_brackets():
    smiles=load_npy(input_path+"smiles")
    protein=load_npy(input_path+"protein")
    label=load_npy(input_path+"label")

    frag_list=[]
    protein_list=[]
    label_list=[]
    for i,s in enumerate(smiles):
        sub=extract_sub(s)
        type_pro=protein2category(protein[i])
        p_frag=to_gram_no_overlap(type_pro)
        frag_list.append(sub)
        protein_list.append(p_frag)
        label_list.append(label[i])
    save_decompose_dataset(input_path,frag_list,protein_list,label_list,fragname)
#     # np.save(input_path+"smiles_"+fragname,frag_list)
#     # np.save(input_path+"protein_category_3_gram_"+fragname,protein_list)
#     # np.save(input_path+"label_"+fragname,label_list)
#
#
# def load_brackets_data(data):
#     f_s=open(input_path+"smiles"+fragname+".txt","w",encoding="utf-8")
#     f_p=open(input_path+"protein"+fragname+".txt","w",encoding="utf-8")
#     s_list=[]
#     p_list=[]
#     l_list=[]
#     for line in data:
#         s,p,l=line.split()
#         f_s.write(s+"\n")
#         f_p.write(p+"\n")
#         s_list.append(s)
#         p_list.append(p)
#         l_list.append(l)
#         # break
#     f_s.close()
#     f_p.close()
#     print(len(s_list),len(p_list),len(l_list))
#     save_decompose_dataset(input_path,s_list,p_list,l_list,"")
#     # np.save(input_path+"smiles",s_list)
#     # np.save(input_path+"protein",p_list)
#     # np.save(input_path+"label",l_list)
#
# def esm_vec(protein):
#     # import esm
#     # Load ESM-1b model
#     model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#     batch_converter = alphabet.get_batch_converter()
#
#     data = [
#         ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#         ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#         ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#         ("protein3", "K A <mask> I S Q"),
#     ]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#
#     # Extract per-residue representations (on CPU)
#     with torch.no_grad():
#         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
#     token_representations = results["representations"][33]
#
#     # Generate per-sequence representations via averaging
#     # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
#     sequence_representations = []
#     for i, (_, seq) in enumerate(data):
#         sequence_representations.append(token_representations[i, 1: len(seq) + 1].mean(0))

def decompose_both_smiles_protein():
    smiles = load_npy(input_path + "smiles")
    protein = load_npy(input_path + "protein")
    label = load_npy(input_path + "label")
    protein_smiles_dict=load_json(input_path+"protein2smiles")

    frag_list = []
    protein_list = []
    protein_raw_list=[]
    label_list = []
    for i, s in enumerate(smiles):
        print(input_path,i,len(smiles))
        if protein[i] not in protein_smiles_dict.keys() or protein_smiles_dict[protein[i]]=="":
            continue
        p=protein_smiles_dict[protein[i]]
        s_sub = extract_sub(s)
        p_sub=extract_sub(p)
        # type_pro = protein2category(protein[i])
        # p_frag = to_gram_no_overlap(type_pro)
        frag_list.append(s_sub)
        protein_list.append(p_sub)
        protein_raw_list.append(protein[i])
        label_list.append(label[i])
    print(len(frag_list),len(protein_list),len(label_list))
    save_decompose_dataset(input_path, frag_list, [protein_list,protein_raw_list], label_list, "_"+fragname+"_ss",p_ori=True)


if __name__ == '__main__':
    dataset_dir=DTI_dataset_config()
    fragname="brackets"
    s='O=C(C)OC1CCCC1C(=O)O'
    print(extract_sub(s))
    for dir in reversed(dataset_dir["datasetDir"]):
        if "DUDE" in dir or "GPCR" in dir:
            continue
        input_path="../"+dir+"/input/"
        # load_dataset(input_path,fragname)
        decompose_both_smiles_protein()
        print(input_path)
    #     # if "DUDE" not in dir:
    #     #     continue
    #     print(dir)
    #     filter_dp_data=load_txt(input_path+"data_filter_dp")
    #     load_brackets_data(filter_dp_data)
    #     decompose_brackets()
