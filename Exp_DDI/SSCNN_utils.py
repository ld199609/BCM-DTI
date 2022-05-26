import sys
sys.path.append("../")
from utils.utils_helper import *
from utils.file_helper import *
import deepsmiles
dataset_config=DTI_dataset_config()
doc2vec_config=doc2vec_params_config()
word2vec_config=word2vec_config()
import glob

def list2zip(data):
    zipped=[tuple(a) for a in data]
    datasetList=list(zip(*zipped))
    smiles=list(datasetList[0])
    protein=list(datasetList[1])
    label=list(datasetList[2])
    return smiles,protein,label

def assert_dir_exist(x):
    if not os.path.exists(x):
        os.makedirs(x)
def save_best_model(model, model_dir, best_epoch):
    # save parameters of trained model
    # print(model_dir)
    torch.save(model.state_dict(), model_dir + '{}.pkl'.format(best_epoch))
    files = glob.glob(model_dir + '*.pkl')
    # print()
    # delete models saved before
    for file in files:
        tmp = file.split('/')[-1]  # windows:\\  linux: /
        tmp = tmp.split('.')[0]
        # print(tmp)
        epoch_nb = int(tmp)
        if epoch_nb < best_epoch:
            os.remove(file)


def load_bert_data():
    print(dataset_config["datasetDir"])
    for root_path in reversed(dataset_config["datasetDir"]):
        save_path=root_path+"/"+"input"
        print(root_path)
        smiles_recap=list(set([a for s in load_npy(save_path+"/"+"recap_filter") for a in s]))
        smiles_brics=list(set([a for s in load_npy(save_path+"/"+"brics_filter")  for a in s]))
        protein_gram=list(set([a for s in load_npy(save_path+"/"+"3_gram_feat_filter") for a in s]))
        smiles_recap_max=max([len(s) for s in load_npy(save_path+"/"+"recap_filter")])
        smiles_brics_max=max([len(s) for s in load_npy(save_path+"/"+"brics_filter")])
        protein_gram_max=max([len(s) for s in load_npy(save_path+"/"+"3_gram_feat_filter")])
        print(protein_gram_max)
        print(len(smiles_recap),len(smiles_brics),len(protein_gram),smiles_recap_max,smiles_brics_max,protein_gram_max)
        np.save(save_path+"/"+"recap_filter_freg",smiles_recap)
        np.save(save_path+"/"+"brics_filter_freg",smiles_brics)
        np.save(save_path+"/"+"protein_filter_feat_freg",protein_gram)

def smiles2deepsmiles():
    for root_path in reversed(dataset_config["datasetDir"]):
        save_path=root_path+"/"+"original"
        input_path=root_path+"/"+"input"
        dataset=load_txt(save_path+"/"+"data_filter")
        f_dp=open(root_path+"/"+"original"+"/"+"data_filter_dp","w")
        print(save_path+"/"+"data_filter")
        converter=deepsmiles.Converter(rings=True,branches=True)
        # dataset_filter_deep=[]
        dp_s=[]
        dp_p=[]
        dp_l=[]
        for i,line in enumerate(dataset):
            s,p,l=line.strip().split(" ")
            try:
                s_en=converter.encode(s)
                s_de=converter.decode(s_en)
                dp_s.append(s_de)
                dp_p.append(p)
                dp_l.append(int(l))
                f_dp.write(" ".join([s_de,p,l]))
                # dataset_filter_deep.append([s_de,p,l])
            except deepsmiles.DecodeError as e:
                print("DecodeError! Error message was '%s'"%e.message)
        # write2txt(save_path+"/"+"data_filter_dp",dataset_filter_deep)
        np.save(input_path+"/"+"smiles_filter_dp",dp_s)
        np.save(input_path+"/"+"protein_filter_dp",dp_p)
        np.save(input_path+"/"+"label_filter_dp",dp_l)
        f_dp.close()

def load_bert_txt():
    print(dataset_config["datasetDir"])
    for root_path in reversed(dataset_config["datasetDir"]):
        save_path=root_path+"/"+"input"
        print(save_path)
        smiles_recap=list(set([a for s in load_npy(save_path+"/"+"recap_filter_dp") for a in s]))
        smiles_brics=list(set([a for s in load_npy(save_path+"/"+"brics_filter_dp")  for a in s]))
        protein_gram=list(set([a for s in load_npy(save_path+"/"+"3_gram_feat_filter_dp") for a in s]))
        smiles_recap_max=max([len(list(set([a for a in s]))) for s in load_npy(save_path+"/"+"recap_filter_dp")])
        smiles_brics_max=max([len(list(set([a for a in s])))  for s in load_npy(save_path+"/"+"brics_filter_dp")])
        protein_gram_max=max([len(list(set([a for a in s])))  for s in load_npy(save_path+"/"+"3_gram_feat_filter_dp")])
        print(protein_gram_max)
        print(len(smiles_recap),len(smiles_brics),len(protein_gram),smiles_recap_max,smiles_brics_max,protein_gram_max)
        np.save(save_path+"/"+"recap_filter_dict_dp",smiles_recap)
        np.save(save_path+"/"+"brics_filter_dict_dp",smiles_brics)
        np.save(save_path+"/"+"protein_filter_feat_dict_dp",protein_gram)

# smiles,protein的最大长度未去重
# 2184 1804 224 19 11 2332
# ../../data/human
# 1680
# 3269 2658 226 20 21 1680
# ../../data/DAVIS
# 850
# 176 162 225 8 11 850
# ../../data/BIOSNAP
# 1728
# 5466 4063 229 19 17 1728

# smiles,protein的最大长度去重
# 181
# 2184 1804 224 19 11 181
# ../../data/human
# 184
# 3269 2658 226 20 21 184
# ../../data/DAVIS
# 156
# 176 162 225 8 11 156
# ../../data/BIOSNAP
# 184
# 5466 4063 229 19 17 184
from  decompose.decompose_brackets_helper import *

def load_bcm_dataset(input_path,decompose1="bcm",decompose2="category"):
    train_dataset=load_txt(input_path + "/" + "Biosnap_train")
    test_dataset=load_txt(input_path + "/" + "Biosnap_test")


    decompose_dataset_train_smiles1=[]
    decompose_dataset_train_smiles2=[]
    decompose_dataset_train_label=[]

    decompose_dataset_test_smiles1 = []
    decompose_dataset_test_smiles2 = []
    decompose_dataset_test_label=[]

    for line in train_dataset:
        s1,s2,l=line.split()
        tmp1=extract_sub(s1)
        tmp2=extract_sub(s2)
        decompose_dataset_train_smiles1.append(tmp1)
        decompose_dataset_train_smiles2.append(tmp2)
        decompose_dataset_train_label.append(int(float(l)))
    for line in test_dataset:
        s1, s2, l = line.split()
        tmp1 = extract_sub(s1)
        tmp2 = extract_sub(s2)
        decompose_dataset_test_smiles1.append(tmp1)
        decompose_dataset_test_smiles2.append(tmp2)
        decompose_dataset_test_label.append(int(float(l)))

    return decompose_dataset_train_smiles1,decompose_dataset_train_smiles2,decompose_dataset_train_label,decompose_dataset_test_smiles1,decompose_dataset_test_smiles2,decompose_dataset_test_label

def load_bpe_dataset(input_path,decompose1="bpe",decompose2="category"):
    datasetSmiles = load_npy(input_path + "/" + "smiles_"+decompose1)
    datasetProtein = load_npy(input_path + "/" + "protein_"+decompose1)
    datasetLabel = load_npy(input_path + "/" + "label_"+decompose1)

    # decompose_dataset_smiles = []
    decompose_dataset_protein = []

    if decompose2 == "category":
        for p in datasetProtein:
            tmp = protein2category(p)
            tmp_gram = to_gram_no_overlap(tmp)
            decompose_dataset_protein.append(tmp_gram)
    elif decompose2 == "":
        for p in datasetProtein:
            tmp_gram = to_gram_no_overlap(p)
            decompose_dataset_protein.append(tmp_gram)
    return datasetSmiles, decompose_dataset_protein, datasetLabel


def load_rdkit_dataset(input_path,decompose1="recap",decompose2="category"):

    datasetSmiles = load_npy(input_path + "/" + "smiles_" + decompose1)
    datasetProtein = load_npy(input_path + "/" + "protein_" + decompose1)
    datasetLabel = load_npy(input_path + "/" + "label_" + decompose1)

    # decompose_dataset_smiles = []
    decompose_dataset_protein = []

    if decompose2 == "category":
        for p in datasetProtein:
            tmp = protein2category(p)
            tmp_gram = to_gram_no_overlap(tmp)
            decompose_dataset_protein.append(tmp_gram)
    elif decompose2 == "":
        for p in datasetProtein:
            tmp_gram = to_gram_no_overlap(p)
            decompose_dataset_protein.append(tmp_gram)
    return datasetSmiles, decompose_dataset_protein, datasetLabel


if __name__ == '__main__':
    # smiles2deepsmiles()
    load_bert_txt()