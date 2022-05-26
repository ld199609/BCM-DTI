import torch.nn as nn
import torch


if torch.cuda.is_available():
    device = torch.device("cuda")  # "cuda:0"
else:
    device = torch.device("cpu")

class FeatConvolution(nn.Module):
    def __init__(self, channels_list, filter_sizes, pool_method='max',layer_num=3):
        """
        :param channels_list: number of filters list
        :param filter_sizes:  filter length list
        """
        super().__init__()
        # (N,C,D,H,W) = (batch,embed_dim,path_num,path_length, node_num)
        self.layers = nn.ModuleList()
        # print(channels_list)
        layer_num=len(channels_list)-1
        for i in range(layer_num):
            tmp_list=nn.ModuleList()
            tmp_list.append(nn.Conv1d(in_channels=channels_list[i],
                                      out_channels=channels_list[i+1],
                                      kernel_size=3,stride=1,padding=1))
            tmp_list.append(nn.BatchNorm1d(channels_list[i+1]))
            tmp_list.append(nn.ELU())
            self.layers.append(tmp_list)

        if pool_method == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool1d((1))
        elif pool_method == 'avg':
            self.pooling_layer = nn.AdaptiveAvgPool1d((1))

        self.changeLayer = nn.Linear(channels_list[0], 512)

    def forward(self, x):
        out = x
        # print(x)
        # print(x.shape)
        # x = x.permute(0, 2, 1)
        # x = self.changeLayer(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        for i in range(len(self.layers)):
            out = self.layers[i][0](out)
            out = self.layers[i][1](out)
            out = self.layers[i][2](out)
            # print(out.shape)
            # out = x - out
            # x = out
            # out = F.relu(out) # (batch,dim,path_num,path_length,node_num)
            # print(i, out.shape)
        # out=x+out
        # print(out.shape)
        out = self.pooling_layer(out)
        out = torch.squeeze(out)  # (batch,dim,node_num)
        return out

class SDAE(nn.Module):
    def __init__(self,input_dim, ngpu):
        super(SDAE, self).__init__()
        # print(input_dim)
        self.ngpu=ngpu
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ELU(),
            )
        self.main = nn.Sequential(
            self.encoder,
            self.decoder)
    def forward(self, x):
        # print(x.shape)
        output=self.main(x)
        # print(output.shape)
        return output

class SSCNN_DTI(nn.Module):
    def __init__(self,args):
        super(SSCNN_DTI, self).__init__()
        self.input_d_dim=args['input_d_dim']
        self.input_p_dim=args['input_p_dim']

        self.embed_d_size=args["embed_d_size"]
        self.embed_p_size=args["embed_p_size"]
        self.input_d_channel_size=args["d_channel_size"]
        self.input_p_channel_size=args["p_channel_size"]

        self.filter_d_size=args["filter_d_size"]
        self.filter_p_size=args["filter_p_size"]

        self.num_embedding=args['num_embedding']

        self.dropout=args['dropout']

        self.fc_size=args['fc_size']

        self.type=args['type']
        self.n_class=args['n_classes']
        self.smiles_embedding=nn.Embedding(num_embeddings=self.input_d_dim,
                                           embedding_dim= self.embed_d_size)
        self.protein_embedding=nn.Embedding(num_embeddings=self.input_p_dim,
                                            embedding_dim= self.embed_p_size)
        # self.smiles_DAE=SDAE(self.input_d_channel_size[0],ngpu=1)
        # self.protein_DAE=SDAE(self.input_p_channel_size[0],ngpu=1)
        # self.input_d_channel_size[0]=64
        # self.input_p_channel_size[0]=64
        self.smiles_cnn=FeatConvolution(self.input_d_channel_size,self.filter_d_size,layer_num=3)
        self.protein_cnn=FeatConvolution( self.input_p_channel_size,self.filter_p_size)

        self.fc_layer=nn.Sequential(nn.Linear(self.fc_size[0],self.fc_size[1]),
                                     # nn.Tanh(),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout),
                                     nn.Linear(self.fc_size[1],self.fc_size[2]),
                                     # nn.Tanh(),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                     )
        self.final_layer=(torch.nn.Linear(self.fc_size[2], self.n_class))


    def forward(self,smiles,protein):
        # print(smiles.shape,protein.shape)

        # print(type(smiles))
        # smiles=torch.tensor(smiles,dtype=torch.long).to(device)
        # protein=torch.tensor(protein,dtype=torch.long).to(device)
        smiles_embedding=self.smiles_embedding(smiles)
        # print(protein.shape,torch.max(protein),self.input_p_dim,self.embed_p_size)
        protein_embedding=self.protein_embedding(protein)
        # print(smiles_embedding.shape)
        # denoise
        # smiles_embedding_denoise=self.smiles_DAE(smiles_embedding)
        # protein_embedding_denoise=self.protein_DAE(protein_embedding)
        # smiles_cnn=self.smiles_cnn(smiles_embedding_denoise)
        # protein_cnn=self.protein_cnn(protein_embedding_denoise)

        smiles_cnn = self.smiles_cnn(smiles_embedding)
        protein_cnn = self.protein_cnn(protein_embedding)


        # print(smiles_cnn.shape,protein_cnn.shape)
        f=torch.cat((smiles_cnn,protein_cnn),dim=1)
        f_p=self.fc_layer(f)

        if not bool(self.type):
            pre=self.final_layer(f_p)
            pre=torch.sigmoid(pre)
            return pre
