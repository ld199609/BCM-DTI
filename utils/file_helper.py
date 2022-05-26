#-*- coding: UTF-8 -*-
import csv,os,json,sys,torch,glob
from itertools import islice
import numpy as np


def list2zip(data):
    zipped=[tuple(a) for a in data]
    datasetList=list(zip(*zipped))
    smiles=list(datasetList[0])
    protein=list(datasetList[1])
    label=list(datasetList[2])
    return smiles,protein,label

def read_csv(file_path):
    """
    delete first row
    :return:
    """
    with open(file_path, "r") as f:
        lines = csv.reader(f)
        return islice(lines, 1, None)

def load_txt(file_path):
    """
    delete first row
    :return:
    """
    with open(file_path+".txt", "r") as f:
        lines = f.readlines()
        return lines


def assert_dir_exist(x):
    if not os.path.exists(x):
        os.makedirs(x)

def save_best_model(model, model_dir, best_epoch):

    torch.save(model.state_dict(), model_dir + '{}.pkl'.format(best_epoch))
    files = glob.glob(model_dir + '*.pkl')

    for file in files:
        tmp = file.split('/')[-1]  # windows:\\  linux: /
        tmp = tmp.split('.')[0]
        epoch_nb = int(tmp)
        if epoch_nb < best_epoch:
            os.remove(file)


def save_decompose_dataset(path,s,p,l,fragname,p_n_gram=False,p_category=False,p_frag=False,p_ori=False):
    np.save(path+"smiles"+fragname,s)
    # if fragname:
    #     link_ch="_"
    # else:
    if p_n_gram:
        np.save(path+"protein_category_3_gram"+fragname,p[2])
    if p_category:
        np.save(path+"protein_category"+fragname,p[1])
    if p_ori:
        np.save(path+"protein"+fragname+"_raw",p[1])
    if p_frag:
        np.save(path+"protein"+fragname,p[0])
    np.save(path+"label"+fragname,l)

def load_dataset(input_path, fragname):
    filter_dp_data = load_txt(input_path + "data_filter_dp")
    f_s = open(input_path + "smiles" + fragname + ".txt", "w", encoding="utf-8")
    f_p = open(input_path + "protein" + fragname + ".txt", "w", encoding="utf-8")
    s_list = []
    p_list = []
    l_list = []
    for line in filter_dp_data:
        s, p, l = line.split()
        f_s.write(s + "\n")
        f_p.write(p + "\n")
        s_list.append(s)
        p_list.append(p)
        l_list.append(l)
    f_s.close()
    f_p.close()
    save_decompose_dataset(input_path, s_list, [p_list], l_list, "",p_ori=True)

def load_tensor(file_name, dtype):
    if "protein" in file_name:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]
    else:
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def write2txt(file_name,list):
    with open(file_name+".txt","a+") as f:
        for i,data in enumerate(list):
            f.write(" ".join(data)+"\n")
    f.close()

def load_col_txt(file_path,col):
    with open(file_path+".txt", "r") as f:
        lines = f.readlines()
        res=[]
        for line in lines:
            a = line.strip().split(" ")
            res.append(a[col-1])
        # print(res[:5])
        return res
def random_dataset_split(alldataset,train_size,valid_size, test_size):
    train_dataset,valid_dataset, test_dataset = torch.utils.data.random_split(alldataset, [train_size,valid_size, test_size])
    return train_dataset,valid_dataset, test_dataset

