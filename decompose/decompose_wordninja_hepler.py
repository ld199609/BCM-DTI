from utils.utils_helper import *
import wordninja

def generate_wordnijia_dataset(filter_dp_data):
    s_wordnijia_list = []
    p_wordnijia_list = []
    p_category_wordnijia_list=[]
    p_category_3_gram_list=[]

    label_list = []

    s_wordnijia_set=[]
    p_wordnijia_set=[]
    p_category_3_gram_set=[]
    p_category_wordnijia_set=[]
    # p_wordnijia_set=[]

    for line in filter_dp_data:
        s, p, l = line.strip().split(" ")
        p_category = protein2category(p)
        p_category_3_gram = to_gram_no_overlap(p_category)
        s_wordnijia=wordninja.split(s)
        p_category_wordnijia=wordninja.split(p_category)
        p_wordnijia=wordninja.split(p)

        s_wordnijia_set.extend([l for l in s_wordnijia])
        p_wordnijia_set.extend([l for l in p_wordnijia])
        p_category_wordnijia_set.extend([l for l in p_category_wordnijia])
        p_category_3_gram_set.extend([l for l in p_category_3_gram])

        s_wordnijia_list.append(s_wordnijia)
        p_wordnijia_list.append(p_wordnijia)
        p_category_wordnijia_list.append(p_category_wordnijia)
        p_category_3_gram_list.append(p_category_3_gram)
        label_list.append(l)

    np.save(input_path + "smiles_" + fragname, s_wordnijia_list)
    np.save(input_path + "protein_" + fragname, p_wordnijia_list)
    np.save(input_path+"protein_category_"+fragname,p_category_wordnijia_list)
    np.save(input_path+"protein_category_3_gram",p_category_3_gram_list)
    np.save(input_path + "label_" + fragname, label_list)

    np.save(input_path+"smiles_"+fragname+"_set",list(set(s_wordnijia_set)))
    np.save(input_path + "protein_" + fragname+"_set", list(set(p_wordnijia_set)))
    np.save(input_path+"protein_category_"+fragname+"_set",list(set(p_category_wordnijia_set)))
    np.save(input_path+"protein_category_3_gram_"+"set",list(set(p_category_3_gram_set)))



if __name__ == '__main__':
    dataset_dir=DTI_dataset_config()
    fragname="wordnijia"
    for dir in reversed(dataset_dir["datasetDir"]):
        input_path=dir+"/input/"
        if "DUDE"  in dir :
            continue
        filter_dp_data=load_txt(input_path+"data_filter_dp")
        print(dir,len(filter_dp_data))
        generate_wordnijia_dataset(filter_dp_data)
