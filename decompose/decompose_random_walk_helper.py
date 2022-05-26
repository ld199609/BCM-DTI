import copy
from scipy.sparse import lil_matrix
import numpy as np
import random
from textrank4zh import TextRank4Sentence
from tqdm import tqdm
import nltk
from utils.utils_helper import *
def generate_random_walk(begin_node, path_length, drug_graph):
    """
    :param begin_node: node id of start node
    :param path_length: sequence length
    :return: node id sequence
    """
    walk = [begin_node]
    while len(walk) < path_length:
        cur = walk[-1]
        cur_neighbors = [n for n in drug_graph.neighbors(cur)]  # 获取当前节点的邻居节点
        if len(cur_neighbors) == 0:  # for some SMILE graph with only few nodes and isolated nodes
            next_node = cur
        else:
            next_node = random.choice(cur_neighbors)
        walk.append(next_node)
    return walk


def generate_random_walk_new(begin_node, p_length, graph, p_num):
    """
    :param p_num:
    :param graph:
    :param p_length:
    :param begin_node: node id of start node
    :return: node id sequence
    每次从当前结点开始随机获得p_length长度的邻居结点（共p_num次）
    """
    final_lists = []
    for num in range(p_num):
        walk = [begin_node]
        while len(walk) < p_length:
            cur = walk[-1]
            cur_neighbors = [n for n in graph.neighbors(cur)]  # 获取当前节点的邻居节点
            if len(cur_neighbors) == 0:  # for some SMILE graph with only few nodes and isolated nodes
                next_node = cur
            else:
                next_node = random.choice(cur_neighbors)
            walk.append(next_node)
        final_lists.append(walk)
    return final_lists


def generate_anonym_walks(length):
    """
    recursive function to generate all anonymous sequence of this length.
    :param length: length of anonymous walker
    :return:  list which includes many lists,each is anonymous sequence
    """
    anonymous_walks = []

    def generate_anonymous_walk(totlen, pre):  # inner function definition
        if len(pre) == totlen:
            anonymous_walks.append(pre)
            return
        else:
            candidate = max(pre) + 1
            for i in range(1, candidate + 1):
                if i != pre[-1]:
                    npre = copy.deepcopy(pre)
                    npre.append(i)
                    generate_anonymous_walk(totlen, npre)

    generate_anonymous_walk(length, [1])
    return anonymous_walks


# 按道理图中不应该出现孤立的节点，但是SMILE字符串转换后得到的确实有孤立的节点
# 且对孤立节点进行随机游走的得到的就是全是１的匿名序列
def generate_walk2num_dict(length):
    anonym_walks = generate_anonym_walks(length)
    anonym_dict = dict()
    tmp_list = [1 for i in range(length)]  # 孤立节点的随机游走序列都是1,其映射的类型为0
    isolated_pattern = "".join([str(x) for x in tmp_list])
    anonym_dict[isolated_pattern] = 0  # for isolated SMILE nodes and padding
    curid = 1
    for walk in anonym_walks:
        swalk = "".join([str(x) for x in walk])  # int list to string
        anonym_dict[swalk] = curid
        curid += 1
    return anonym_dict


# 虽有游走序列转换为匿名随机游走序列
def to_anonym_walk(walk):
    """

    :param walk: node id sequcence list
    :return: annoym_walk: int sequence list
    """
    num_app = 0
    apped = dict()  # save the node id first appear
    anonym = []
    for node in walk:
        if node not in apped:
            num_app += 1
            apped[node] = num_app
        anonym.append(apped[node])
    return anonym


# 紧密中心性的计算　（v-1）/（到其他节点距离的总和）　＝　某个节点到其他节点距离的平均值的倒数
def closeness_centrality_calculation(graph, max_num=10):
    """
    :param graph: networkx graph
    :param max_num: the number to be selected which has the biggest closeness centrality calculation
    :return: node id list
    """
    nodes_num = len(graph.nodes())
    centrality_list = []
    for src_id in range(nodes_num):
        shortest_dic = nx.shortest_path_length(graph, source=src_id)
        distance_sum = 0
        for target_id in range(nodes_num):
            if target_id in shortest_dic:  # 转换后的分子图中可能存在孤立的子图,因此需要判断一下,原因待探究
                distance_sum += shortest_dic[target_id]
        if distance_sum == 0:  # the isolated nodes.
            centrality = 0
        else:
            centrality = (nodes_num - 1) / distance_sum
        centrality_list.append(centrality)
    tmp = zip(range(nodes_num), centrality_list)
    sorted_tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    res_id_list = []
    # when the whole node number is less than max_num, do random sample.
    # [(1, 1.0), (0, 0.5714285714285714), (2, 0.5714285714285714), (3, 0.5714285714285714), (4, 0.5714285714285714)]
    # print(sorted_tmp)  # [tuple1,tuple2,...]  tuple1:(node_id,centrality_calc)
    current_node_num = len(sorted_tmp)
    used_node_num = min(current_node_num, max_num)  # 实际选择出来的节点
    selected_node_num = []
    for t in range(used_node_num):
        res_id_list.append(sorted_tmp[t][0])
        selected_node_num.append(sorted_tmp[t][0])
    remain_node_num = max_num - used_node_num
    # 当节点数目不够时,是否进行重复选择
    # if remain_node_num > 0:
    #     repeat_node_list = np.random.choice(selected_node_num, size=remain_node_num, replace=True)
    #     res_id_list.extend(repeat_node_list)
    # assert len(res_id_list) == max_num
    return res_id_list


# the most important function
def generate_graph_data(adj_numpy_matrix, feat_graph, str_to_index_dic, num_paths=50, path_length=6, max_num=10):
    """
    :param str_to_index_dic: 匿名随机游走序列映射字典.  key:匿名随机游走字符串 value:匿名随机游走类型编号,也就是所谓的结构模编号
    :param adj_numpy_matrix: (node_num,node_num) 传入的特征矩阵在做预处理的时候就是对称矩阵,没有self-loop
    :param feat_graph: (node_num,feat_dim)
    :param num_paths: 50
    :param path_length: 6
    :param max_num: 10
    :return: numpy
            (10,)             selected nodes list
            (10, 52)          graph overall features  (removed)
            (10, 50)          graph structure pattern representation
            (10, 50, 6, 38)   graph walk features representation (removed)
    """
    # 01 使用networksx将numpy的邻接矩阵建立图
    mm = lil_matrix(adj_numpy_matrix)
    real_node_num = mm.shape[0]
    # print("the node number of real graph %d" % real_node_num)
    rows = mm.rows
    adjacency_dic = dict(zip([i for i in range(rows.shape[0])], rows))
    # print(adjacency_dic)
    edge_list = []
    for one_edge in adjacency_dic.keys():
        for another_edge in adjacency_dic[one_edge]:
            edge_list.append((one_edge, another_edge))
    # print(edge_list)
    drug_graph = nx.Graph(edge_list)
    # the smile graph may have some isolated nodes,this step help adding isolated nodes
    for node_id in range(real_node_num):
        drug_graph.add_node(node_id)

    # 02 获取需要随机游走的节点编号
    selected_node_list = closeness_centrality_calculation(graph=drug_graph, max_num=max_num)
    # print("The selected nodes ids for random walk:" + str(selected_node_list))

    # 03 获取节点id到随机游走序列的映射
    walk_dic = {}  # key:node_id, value: walk list
    for selected_id in selected_node_list:
        tmp_walk_lists = generate_random_walk_new(begin_node=selected_id, p_length=path_length, graph=drug_graph,
                                                  p_num=num_paths)
        walk_dic[selected_id] = tmp_walk_lists  # (path_num,path_length)
    # print("random walker for each selected nodes" + str(walk_dic))

    # 04 将每个节点获得的匿名随机游走序列转化得到(node_num,dim)维度的特征表示
    id_walk_feat_dic = {}
    node_feat_dim = feat_graph.shape[1]
    for selected_id in walk_dic.keys():  # 遍历选择出来的节点

        passed_node_lists = walk_dic[selected_id]
        feat_array = np.zeros(shape=(num_paths, path_length, node_feat_dim))  # 每个节点的特征维度是(路径数,路径长度,节点的维度)
        for path_idx, tmp_list in enumerate(passed_node_lists):  # 遍历该节点的每一个路径
            feat_array[path_idx, :, :] = feat_graph[tmp_list]    # 使用list作为作为索引返回新的array,要求list的范围不超过array维度
        id_walk_feat_dic[selected_id] = feat_array  # (path_number,path_length,feat_dim)
    # print("Graph walker feature:" + str(id_walk_feat_dic))

    # 05 将随机游走序列转换为匿名随机游走序列
    anonymous_walk_dic = {}
    for key, value in walk_dic.items():
        tmp_list = []
        for id_list in walk_dic[key]:
            tmp_list.append(to_anonym_walk(id_list))
        anonymous_walk_dic[key] = tmp_list
    # 匿名随机游走序列都是1代表这是一个孤立的节点
    # print("anonymous walker for each selected nodes" + str(anonymous_walk_dic))

    # 0表示孤立的节点, 1~n分别是对一种匿名随机游走序列进行映射.
    # 06 将随机游走序列映射为结构模式索引(节点的特征该如何编码)
    # 对匿名随机游走表示的结果进行embedding嵌入,从而让各个分子之间共享相同的嵌入表示
    # str_to_index_dic = generate_walk2num_dict(path_length)
    type_walk_dic = {}  # key: selected node id  value: walker type list
    for key, value in anonymous_walk_dic.items():
        tmp_type_list = []
        for id_list in anonymous_walk_dic[key]:
            str_walk = "".join(str(x) for x in id_list)
            tmp_type_list.append(str_to_index_dic[str_walk])
        type_walk_dic[key] = tmp_type_list
    # print("Graph structure pattern:" + str(type_walk_dic))

    # 07 得到该图的   结构模式向量+随机游走序列的特征表示
    # 比较重要的数据结构:
    # type_walk_dic:      key是目标节点id,value是该目标节点周围的结构模型list,长度为path_number
    # id_walk_feat_dic:   key是目标节点,value是目标节点的随机游走序列特征表示,长度为(path_number,path_length,feat_dim)
    # selected_node_list: 该分子图的目标节点id列表
    final_graph_walk_array = np.zeros(shape=(max_num, num_paths))
    final_feat_walk_array = np.zeros(shape=(max_num, num_paths, path_length, node_feat_dim))
    for idx, tmp_id in enumerate(selected_node_list):
        final_graph_walk_array[idx, :] = type_walk_dic[tmp_id]
        final_feat_walk_array[idx, :, :, :] = id_walk_feat_dic[tmp_id]
    # print(len(selected_node_list),len(final_feat_walk_array),len(final_graph_walk_array))
    # 返回了选择的的nodeid list ,最后的随机node特征，
    # return [selected_node_list, final_feat_walk_array, final_graph_walk_array]
    return final_feat_walk_array, final_graph_walk_array


# def make_graph_data_list(adj_p, feat_p, num_p=50, path_l=6, max_n=10):
#     print("Path number %d,Path length %d,Maximum Node number %d" % (num_p, path_l, max_n))
#     adj_matrix_list = load_decompressed_pzip(adj_p)  # list  <class 'numpy.ndarray'> (node_num,node_num)
#     feat_matrix_list = load_decompressed_pzip(feat_p)  # list  <class 'numpy.matrix'> (node_num,feat_dim)
#     graph_number = len(feat_matrix_list)
#     str_to_index_dic = generate_walk2num_dict(path_l)  # structure pattern map dictionary
#     extraction_list = []
#     for i in tqdm(range(graph_number)):
#         tmp_data1,tmp_data2 = generate_graph_data(adj_matrix_list[i], feat_matrix_list[i], str_to_index_dic=str_to_index_dic,
#                                        num_paths=num_p, path_length=path_l, max_num=max_n)
#         extraction_list.append(tmp_data1)
#
#     return extraction_list


def make_graph_data_list_called(adj_matrix_list, feat_matrix_list, num_p=50, path_l=6, max_n=10):
    # adj_matrix_list = load_decompressed_pzip(adj_p)  # list  <class 'numpy.ndarray'> (node_num,node_num)
    # feat_matrix_list = load_decompressed_pzip(feat_p)  # list  <class 'numpy.matrix'> (node_num,feat_dim)
    print("Path number %d,Path length %d,Maximum Node number %d" % (num_p, path_l, max_n))
    graph_number = len(feat_matrix_list)
    str_to_index_dic = generate_walk2num_dict(path_l)  # structure pattern map dictionary
    extraction_list = []
    struct_list=[]
    graph_struct_list=[]
    # 对每个分子的图进行随机游走获得子序列
    for i in tqdm(range(graph_number)):
        tmp_data1, tmp_data2= generate_graph_data(adj_matrix_list[i], feat_matrix_list[i], str_to_index_dic=str_to_index_dic,
                                       num_paths=num_p, path_length=path_l, max_num=max_n)
        struct_list.append(tmp_data1)
        graph_struct_list.append(tmp_data2)
    return struct_list,graph_struct_list,len(str_to_index_dic)

def generate_subsequence_data(mol,adj,str_to_index_dic, num_paths=50, path_length=6, max_num=712):
    mm = lil_matrix(adj)
    real_node_num = mm.shape[0]
    subsquence_list=[]
    # 01 使用networksx将numpy的邻接矩阵建立图
    # print("the node number of real graph %d" % real_node_num)
    rows = mm.rows
    adjacency_dic = dict(zip([i for i in range(rows.shape[0])], rows))
    # print(adjacency_dic)
    edge_list = []
    for one_edge in adjacency_dic.keys():
        for another_edge in adjacency_dic[one_edge]:
            edge_list.append((one_edge, another_edge))
    # print(edge_list)
    drug_graph = nx.Graph(edge_list)
    # the smile graph may have some isolated nodes,this step help adding isolated nodes
    for node_id in range(real_node_num):
        drug_graph.add_node(node_id)

    # 02 获取需要随机游走的节点编号
    selected_node_list = closeness_centrality_calculation(graph=drug_graph, max_num=max_num)
    # print("The selected nodes ids for random walk:" + str(selected_node_list))

    # 03 获取节点id到随机游走序列的映射
    walk_dic = {}  # key:node_id, value: walk list
    for selected_id in selected_node_list:
        tmp_walk_lists = generate_random_walk_new(begin_node=selected_id, p_length=path_length, graph=drug_graph,
                                                  p_num=num_paths)
        walk_dic[selected_id] = tmp_walk_lists  # (path_num,path_length)
    # print("random walker for each selected nodes" + str(walk_dic))

    # 04 将每个节点获得的匿名随机游走序列转化得到(node_num,dim)维度的特征表示
    id_walk_feat_dic = {}
    for selected_id in walk_dic.keys():  # 遍历选择出来的节点
        passed_node_lists = walk_dic[selected_id]
        tmp_sub_list=[]
        for path_idx, tmp_list in enumerate(passed_node_lists):
            tmp_sub=''
            # 遍历该节点的每一个路径
            for idx in tmp_list:
                tmp_sub=tmp_sub+mol.GetAtomWithIdx(idx).GetSmarts()
            tmp_sub_list.append(tmp_sub)
        subsquence_list.append(" ".join(list(set(tmp_sub_list)))+".")

    # print(len())
    return "\n".join(subsquence_list)
           # print(tmp_list)
        # print("Graph walker feature:" + str(id_walk_feat_dic))
    # the smile graph may have some isolated nodes,this step help adding isolated nodes

def text_abstract(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    for item in tr4s.get_key_sentences(num=1):
        return item.sentence

def load_random_walk(smiles_all,protein_all,label_all,path_l=6,num_p=50,max_n=712):
    str_to_index_dic = generate_walk2num_dict(path_l)  # structure pattern map dictionary
    subsquence_list_all=[]
    k_gram_no_overlap_list=[]
    label_list=[]
    k_gram_set=[]
    frag_set=[]
    for i,suff in enumerate(smiles_all):
        # try:
            # if i%1000==0:
            #     print(i,len(smiles_all))
            # mol=Chem.MolFromSmiles(smiles_all[i])
            # adj=adjacent_matrix(mol)
            # tmp_subsquence_list=generate_subsequence_data(mol,adj,str_to_index_dic=str_to_index_dic,
            #                                num_paths=num_p, path_length=path_l, max_num=max_n)
            # # print(smiles_all[i],tmp_subsquence_list)
            # text_extract=text_abstract(tmp_subsquence_list).replace(".","").strip().split(" ")
            # print(text_extract)
            token_list=nltk.word_tokenize(smiles_all[i])
            protein_category = protein2category(protein_all[i])
            k_gram_no_overlap = to_gram_no_overlap(protein_category)
            k_gram_no_overlap_list.append(k_gram_no_overlap)
            frag_set.extend([l for s in token_list for l in s])
            k_gram_set.extend([l for l in k_gram_no_overlap ])
            subsquence_list_all.append(token_list)
            label_list.append(label_all[i])
        # except:
        #     print(smiles_all[i])

    return subsquence_list_all,frag_set,k_gram_no_overlap_list,k_gram_set,label_list

def protein_process(protein_list_all):
    k_gram_no_overlap_list=[]
    for protein in protein_list_all:
        protein_category=protein2category(protein)
        k_gram_no_overlap=to_gram_no_overlap(protein_category)
        k_gram_no_overlap_list.append(k_gram_no_overlap)
    k_gram_category_set=set([s for l in k_gram_no_overlap_list for s in l])
    return k_gram_no_overlap_list,k_gram_category_set

if __name__ == '__main__':
    dataset_dir=DTI_dataset_config()
    fragname="random_walk"
    frag_smiles=[]
    # frag_set=[]
    for dir in reversed(dataset_dir["datasetDir"]):
        input_path=dir+"/input/"
        if "celegans" not  in dir:
            continue
        print(dir)

        smiles_filter_dp_data=load_col_txt(input_path+"data_filter_dp",1)
        protein_filter_dp_data=load_col_txt(input_path+"data_filter_dp",2)
        label_filter_dp_data=load_col_txt(input_path+"data_filter_dp",3)

        # protein_category_3_gram,protein_category_3_gram_set=protein_process(protein_filter_dp_data)

        sub_sesquence,frag_set,protein_category_3_gram,protein_category_3_gram_set,label_list=load_random_walk(smiles_filter_dp_data,protein_filter_dp_data,label_filter_dp_data)
        # print(frag_set)
        print(len(sub_sesquence),len(protein_category_3_gram),len(label_list))
        np.save(input_path+"smiles_"+fragname,sub_sesquence)
        np.save(input_path+"protein_category_3_gram_"+fragname,protein_category_3_gram)
        np.save(input_path+"label_"+fragname,label_list)

        np.save(input_path+"smiles_"+fragname+"_set",list(set(frag_set)))
        np.save(input_path+"protein_"+fragname+"_set",list(set(protein_category_3_gram_set)))
        # print()




