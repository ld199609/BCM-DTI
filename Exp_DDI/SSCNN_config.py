def SSCNN_args():
    config = {
             'embed_d_size': 128,
              'embed_p_size': 128,
              'd_channel_size': [64,512,512, 512],
              'p_channel_size': [181,512,512,512],
              'filter_d_size': [32,32, 32,32],
              'filter_p_size': [32,32, 64],
              'batch_size': 32,
              'epochs': 100,
              'num_embedding': 32,
              'dropout': 0.5,
              'fc_size': [1024, 1024, 512],
              'lr':1e-5,
              'type':0,
              'n_classes':1,
              'clip':True,
                'stop_counter':30,
              }
    config['max_drug_seq'] = {"celegans": [19, 11], "human": [20, 21], "BIOSNAP": [19, 17], "DAVIS": [8, 11],
                              "BindingDB": [68, 23]}
    # 去重复
    config['max_protein_seq'] = {"celegans": 181, "human": 184, "BIOSNAP": 184, "DAVIS": 156, "BindingDB": 173}
    # 未去重复
    # config['max_protein_seq'] = {"celegans": 2332, "human": 1680, "BIOSNAP": 184, "DAVIS": 156, "BindingDB": 1}

    config['input_d_dim']={"celegans":[2184,1804],"human":[3269,2658],"BIOSNAP":[5733,4269],"DAVIS":[184,168],"BindingDB":[6631,3910]}
    config['input_p_dim']={"celegans":224,"human":226,"BIOSNAP":229,"DAVIS":225,"BindingDB":225}
    return config
