import torch.nn as nn
import torch

class BasicConvResBlock(nn.Module):

    def __init__(self, input_dim=128, n_filters=256, kernel_size=3, padding=1, stride=1, shortcut=True, downsample=None):
        super(BasicConvResBlock, self).__init__()

        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(input_dim, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(n_filters)

        self.leyer=nn.Linear(128,256)

    def forward(self, x):
        residual = x
        residual=residual.permute(0, 2, 1)
        print(residual.shape)
        residual=self.leyer(residual)
        print(x.shape,residual.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        print(out.shape)
        residual = residual.permute(0, 2, 1)
        if self.shortcut:
            out -= residual

        out = self.relu(out)

        return out
class FeatConvolution(nn.Module):
    def __init__(self, channels_list, filter_sizes, pool_method='max',layer_num=3):
        """
        :param channels_list: number of filters list
        :param filter_sizes:  filter length list
        """
        super().__init__()
        # (N,C,D,H,W) = (batch,embed_dim,path_num,path_length, node_num)
        self.layers = nn.ModuleList()
        # tmp=0
        # print(channels_list)
        for i in range(layer_num):
            tmp_list=nn.ModuleList()
            tmp_list.append(nn.Conv1d(in_channels=channels_list[i],
                                      out_channels=512,
                                      kernel_size=3,stride=1,padding=1))
            tmp_list.append(nn.BatchNorm1d(channels_list[i+1]))
            tmp_list.append(nn.ELU())
            # channels_list[i+1]=128
            self.layers.append(tmp_list)
            tmp=i
        # self.layers.append(nn.Conv1d(in_channels=channels_list[i]))
        if pool_method == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool1d((1))
        elif pool_method == 'avg':
            self.pooling_layer = nn.AdaptiveAvgPool1d((1))
        # print(channels_list[0],channels_list[tmp])
        self.changeLayer=nn.Linear(channels_list[0],512)


    def forward(self, x):
        out = x
        # print(x.shape)
        x=x.permute(0,2,1)
        x=self.changeLayer(x)
        x = x.permute(0, 2, 1)
        for i in range(len(self.layers)):
            out=self.layers[i][0](out)
            out=self.layers[i][1](out)
            out=self.layers[i][2](out)
            out = x - out
            x=out
            # out = F.relu(out) # (batch,dim,path_num,path_length,node_num)
            # print(i, out.shape)
        # out=x-out
        out = self.pooling_layer(out)
        out = torch.squeeze(out)  # (batch,dim,node_num)
        return out

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
        self.smiles1_embedding=nn.Embedding(num_embeddings=self.input_d_dim,
                                           embedding_dim= self.embed_d_size)
        self.smiles2_embedding=nn.Embedding(num_embeddings=self.input_d_dim,
                                            embedding_dim= self.embed_d_size)

        self.smiles1_cnn=FeatConvolution(self.input_d_channel_size,self.filter_d_size)
        self.smiles2_cnn=FeatConvolution( self.input_d_channel_size,self.filter_d_size)

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


    def forward(self,smiles1,finger,smiles2):
        smiles1_embedding=self.smiles1_embedding(smiles1)
        smiles2_embedding=self.smiles2_embedding(smiles2)

        smiles1_cnn=self.smiles1_cnn(smiles1_embedding)
        # print(smiles1_cnn.shape)
        smiles2_cnn=self.smiles2_cnn(smiles2_embedding)
        # print(smiles_cnn.shape,protein_cnn.shape)
        f=torch.cat((smiles1_cnn,smiles2_cnn),dim=1)
        # print(f.shape)
        f_p=self.fc_layer(f)

        if not bool(self.type):
            pre=self.final_layer(f_p)
            pre=torch.sigmoid(pre)
            return pre
# if __name__ == '__main__':
    # c = torch.randint(1, 10, [256, 128, 3])
    # c=torch.FloatTensor(256, 128, 3)
    #
    # model=BasicConvResBlock()
    # model.forward(c)
