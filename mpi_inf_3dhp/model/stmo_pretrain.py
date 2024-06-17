import torch
import torch.nn as nn
from model.block.vanilla_transformer_encoder_pretrain import Transformer, Transformer_dec
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        #self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w1 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        #self.w2 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Conv1d(self.l_size, self.l_size, kernel_size=1)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class FCBlock(nn.Module):

    def __init__(self, channel_in, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.channel_in = channel_in
        self.stage_num = 3
        self.p_dropout = 0.1
        #self.fc_1 = nn.Linear(self.channel_in, self.linear_size)
        self.fc_1 = nn.Conv1d(self.channel_in, self.linear_size, kernel_size=1)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        #self.fc_2 = nn.Linear(self.linear_size, channel_out)
        self.fc_2 = nn.Conv1d(self.linear_size, channel_out, kernel_size=1)

        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)

        return x

class Model_MAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
        stride_num = args.stride_num
        self.spatial_mask_num = args.spatial_mask_num
        self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.length = length
        dec_dim_shrink = 2

        self.encoder = FCBlock(2*self.num_joints_in, channel, 2*channel, 1)

        self.Transformer = Transformer(layers, channel, d_hid, length=length)
        self.Transformer_dec = Transformer_dec(layers-1, channel//dec_dim_shrink, d_hid//dec_dim_shrink, length=length)

        self.encoder_to_decoder = nn.Linear(channel, channel//dec_dim_shrink, bias=False)
        self.encoder_LN = LayerNorm(channel)
        
        self.fcn_dec = nn.Sequential(
            nn.BatchNorm1d(channel//dec_dim_shrink, momentum=0.1),
            nn.Conv1d(channel//dec_dim_shrink, 2*self.num_joints_out, kernel_size=1)
        )

        # self.fcn_1 = nn.Sequential(
        #     nn.BatchNorm1d(channel, momentum=0.1),
        #     nn.Conv1d(channel, 3*self.num_joints_out, kernel_size=1)
        # )

        self.dec_pos_embedding = nn.Parameter(torch.randn(1, length, channel//dec_dim_shrink))
        self.mask_token = nn.Parameter(torch.randn(1, 1, channel//dec_dim_shrink))

        self.spatial_mask_token = nn.Parameter(torch.randn(1, 1, 2))

    def forward(self, x_in, mask, spatial_mask):
        x_in = x_in[:, :, :, :, 0].permute(0, 2, 3, 1).contiguous()
        b,f,_,_ = x_in.shape

        # spatial mask out
        x = x_in.clone()

        x[:,spatial_mask] = self.spatial_mask_token.expand(b,self.spatial_mask_num*f,2)


        x = x.view(b, f, -1)

        x = x.permute(0, 2, 1).contiguous()

        x = self.encoder(x)

        x = x.permute(0, 2, 1).contiguous()
        feas = self.Transformer(x, mask_MAE=mask)

        feas = self.encoder_LN(feas)
        feas = self.encoder_to_decoder(feas)

        B, N, C = feas.shape

        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.dec_pos_embedding.expand(B, -1, -1).clone()
        pos_emd_vis = expand_pos_embed[:, ~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[:, mask].reshape(B, -1, C)
        x_full = torch.cat([feas + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        x_out = self.Transformer_dec(x_full, pos_emd_mask.shape[1])

        x_out = x_out.permute(0, 2, 1).contiguous()
        x_out = self.fcn_dec(x_out)

        x_out = x_out.view(b, self.num_joints_out, 2, -1)
        x_out = x_out.permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-1)
        
        return x_out




