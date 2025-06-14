# This file is library of Airfoil_DDPM.
# Author: Zhe Wen
# Date: 2025-5-22
# Copyright (c) Zhejiang University. All rights reserved.
# See LICENSE file in the project root for license information.

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module
import torch.nn.functional as F
from math import sqrt
import csv
from einops import rearrange, repeat
import numpy as np
from torch.nn import MultiheadAttention

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def sin_cos_embedding(pc, embed_dim=32, scale=1.0):
    """
    pc: [B, D, N] (D通常为3)
    embed_dim: 每个坐标维度的编码维度（总维度=D*embed_dim*2）
    scale: 控制频率范围（默认为1.0）
    """
    B, D, N = pc.shape
    pos = pc.permute(0, 2, 1)  # [B, N, D]
    
    # 生成频率因子
    dim_t = torch.arange(embed_dim, dtype=torch.float32, device=pc.device)
    dim_t = scale ** (2 * (dim_t // 2) / embed_dim)  # [embed_dim]
    
    # 计算正弦和余弦编码
    pos_enc = pos.unsqueeze(-1) * dim_t.view(1, 1, 1, -1)  # [B, N, D, embed_dim]
    pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)  # [B, N, D, 2*embed_dim]
    
    # 展平并拼接原始特征
    pos_enc = pos_enc.reshape(B, N, -1).permute(0, 2, 1)  # [B, D*2*embed_dim, N]
    return pos_enc


class PointNetEncoder(nn.Module):
    def __init__(self, dim=2, dim_out=128):
        ''' 
        Input : [B, dim, num]
        Output : [B, dim, 1]
        '''
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(dim, 64, 1) #输入通道数、输出通道数、卷积核大小
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, dim_out, 1)

    def forward(self, x):
        B, D, N = x.size() #[B, 2, N]
        x = F.relu(self.conv1(x))#[B, 64, N]
        x = F.relu(self.conv2(x))#[B, 128, N]
        x = self.conv3(x)#[B, 1024, N]
        x = torch.max(x, 2, keepdim=True)[0] #沿着#[B, D, N]的第2个维度取maxpooling(N维度),[B, D, 1]
        return x


#残差连接
class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class Block(Module):#包含卷积、归一化、激活和 Dropout 等操作的函数
    ''' 
    Input : [B, dim, num]
    Output : [B, dim, num]
    '''
    def __init__(self, dim, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 1)#一维卷积,实际上是全连接
        self.norm = RMSNorm(dim)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.act(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    '''
    Input : [B, dim, num]
    time_emb : [B, time_emb_dim]
    Output : [B, dim, num]
    '''
    def __init__(self, dim, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dropout = dropout)
        self.block2 = Block(dim)
        self.res_conv = nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):#如果有time embedding，则用MLP对time embedding进行处理
            time_emb = time_emb / 500 
            time_emb = self.mlp(time_emb) #time_emb : [B, dim_in * 2]
            time_emb = rearrange(time_emb, 'b c -> b c 1')#time_emb : [B, dim_in * 2, 1]
            scale_shift = time_emb.chunk(2, dim = 1)#[B, dim_in, 1] * 2

        h = self.block1(x, scale_shift = scale_shift)#用time_emb得到的scale_shift对x进行缩放

        h = self.block2(h)

        return h + self.res_conv(x)#用MLP作为f（x），用res_conv进行维度调整、残差连接

    
class CrossAttention(nn.Module):
    '''
    Input :  query [B, query_dim, num] 来自整体特征
             context [B, context_dim, num] 来自点云
             value [B, value_dim, num] 来自点云
    Output : [B, out_dim, num]
    '''
    def __init__(self, query_dim, out_dim, context_dim=None, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        context_dim = default(context_dim, query_dim)
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.proj_q = nn.Sequential(
            nn.Linear(query_dim, 40, bias=False),
            nn.ReLU(),
            nn.Linear(40, dim_head, bias=False)
        )
        self.proj_k = nn.Sequential(
            nn.Linear(query_dim, 40, bias=False),
            nn.ReLU(),
            nn.Linear(40, dim_head, bias=False)     
        )
        self.proj_v = nn.Sequential(
            nn.Linear(query_dim, 40, bias=False),
            nn.ReLU(),
            nn.Linear(40, dim_head, bias=False)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_head, out_dim),
            nn.Dropout(dropout)
        )

        if context_dim is not None:
            self.proj_c2k = nn.Sequential(
            nn.Linear(context_dim, dim_head, bias=False)
            )
            self.proj_c2v = nn.Sequential(
            nn.Linear(context_dim, dim_head, bias=False)
            )
        self.cross_attn = MultiheadAttention(
            embed_dim=dim_head,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x, context=None, mask=None): # x: [B, query_dim, num]
        x = x.transpose(2, 1)#[B, num, query_dim]
        q = self.proj_q(x)
        
        if context is not None:
            context = context.transpose(2, 1)#[B, num, context_dim]
            #context=context.repeat(1, x.shape[1], 1)
            k = self.proj_c2k(context) # [B, num, inner_dim]
            v = self.proj_c2v(context)
        else:
            k = self.proj_k(x)
            v = self.proj_v(x)#[B, num, inner_dim]

        attn_output, _ = self.cross_attn(
                    query=q,
                    key=k,
                    value=v
                )

        out = self.to_out(attn_output)#[B, num, out_dim]
        return out.transpose(2, 1)#[B, out_dim, num]
    

class UNet_Block(nn.Module):
    '''
    Input:  [B, dim, num]
            [B, time_emb_dim]
    Output: [B, dim, num]
    '''
    def __init__(self, dim, context_dim, time_emb_dim=1,  dropout = 0.):
        super().__init__()
        self.resnet = ResnetBlock(dim=dim, time_emb_dim=time_emb_dim, dropout=dropout)
        self.crossattention = CrossAttention(query_dim=dim, out_dim=dim, context_dim=context_dim, heads=4, dim_head=32, dropout=0.)

    def forward(self, x, time_emb = None, context=None):
        x = self.resnet(x, time_emb)
        x = self.crossattention(x, context=context)
        return x
    

class Unet(nn.Module):
    '''
    Input:  x:[B, num, dim]
            time_emb:[B, time_emb_dim(1)]
            context:[B, num(1), context_dim]
    Output: [B, num_out, dim_out]
    '''
    def __init__(self, dim = 2, encoder_dim=128, time_emb_dim=1, context_dim_1 = 3, context_dim_2 = 3, dropout = 0.):
        super().__init__()
        # self.encoder = PointNetEncoder(dim=dim, dim_out=encoder_dim)
        # self.block_1 = UNet_Block(dim=dim, context_dim=encoder_dim, time_emb_dim=time_emb_dim, dropout = dropout)
        self.block_2 = UNet_Block(dim=dim + dim*2*16, context_dim=context_dim_1, time_emb_dim=time_emb_dim, dropout = dropout)
        self.block_3 = UNet_Block(dim=dim + dim*2*16, context_dim=context_dim_2, time_emb_dim=time_emb_dim, dropout = dropout)

    def forward(self, x, time_emb, context_1=None, context_2=None):
        # position_emb = self.encoder(x)
        # x = self.block_1(x, time_emb=time_emb, context=position_emb)
        embedding = sin_cos_embedding(x, embed_dim=16)  # [B, D * 2 * 16, N]
        x = torch.cat([x, embedding], dim=1)
        x = self.block_2(x, time_emb=time_emb, context=context_1)
        x = self.block_3(x, time_emb=time_emb, context=context_2)
        return x[:, :2, :]
            


    
'''Generation'''
def tensor_combine(tensor1,tensor2):
    combinations1 = []
    combinations2 = []
    if (tensor1 is None)or(tensor2 is None):
        return tensor1,tensor2
    for i in range(tensor1.shape[0]):
        for j in range(tensor2.shape[0]):
            combinations1.append(tensor1[i])
            combinations2.append(tensor2[j])

    return torch.stack(combinations1), torch.stack(combinations2)

def weighted_average(tensor):
    n = tensor.shape[0]
    weights = torch.zeros(n)
    for i in range(n):
        # 以距离的平方的倒数作为权重，距离越近权重越大
        distances = torch.tensor([torch.sum((tensor[i] - tensor[j])**2) for j in range(n)])
        weights[i] = torch.exp(-distances[i])
    # 归一化权重
    weights = weights / torch.sum(weights)
    weights=weights.cuda()
    output=torch.sum(tensor * weights.view(n, 1, 1), dim=0)
    return output.reshape(-1,output.shape[-2],output.shape[-1])

def cal_alpha_bar(alpha, max_t):
    alphabar=torch.empty(max_t)
    temp=1
    for i in range(max_t):
        temp=temp*alpha[i]
        alphabar[i]=temp
    return alphabar

#适用于多目标约束的情况
class Airfoil_DDPM_multitask(nn.Module):
    '''
    model:Unet
    x: init noise
    context_1, context_2 : [num, size]

    '''
    def __init__(self, model, device='cuda'):
        super().__init__()
        #self.model=Unet(input_dim, query_size,context_size, down_sampling_dim, dropout = 0.)
        # 模型加载
        #checkpoint = torch.load(model_filename, map_location=torch.device('cuda'),weights_only=True)   ### 加载神经网络模型
        #self.model.load_state_dict(checkpoint['models'])
        self.model=model.to(device)

        self.beta=torch.linspace(0.0001, 0.02, steps=500)
        self.alpha=1-self.beta
        self.alphabar=cal_alpha_bar(self.alpha, 500)


        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alphabar = self.alphabar.to(device)
        self.model = self.model.to(device)
        self.device = device

    def forward(self, x, context_1, context_2, CFG=1, t_max=500):
        mean = 0.5  # 均值
        std = 0.5  # 标准差
        if x is None:
            x = torch.normal(mean=mean, std=std, size=(200,)).reshape(1,2,-1).to(self.device)
        else:
            xmean = x.mean()
            xstd = x.std()
            x = (x - xmean) / xstd*2

        if context_1 is not None:
            shape_1=context_1.shape[0]
        else:
            shape_1=1
        if context_2 is not None:
            shape_2=context_2.shape[0]
        else:
            shape_2=1
        
        Bs = shape_1*shape_2
        x[:, 1, :] *= 0.1

        x = x.repeat(Bs, 1, 1).to(self.device)
        context_1_combined,context_2_combined = tensor_combine(context_1,context_2)
        for t in range(t_max-1, 0, -1):
            time_embedding = torch.tensor(t, dtype=torch.float32).unsqueeze(0).expand(Bs, -1).to(self.device)
            hidden_without_context=self.model(x, time_embedding, context_1=None, context_2=None)

            if context_1_combined or context_2_combined:
                hidden_with_context=self.model(x, time_embedding, context_1=context_1_combined, context_2=context_2_combined)
                #因为是多目标，所以需要对hidden_with_context进行处理
                hidden_with_context=weighted_average(hidden_with_context)
                noise_pred= hidden_without_context + CFG*( hidden_with_context-hidden_without_context)
            else:
                noise_pred = hidden_without_context
            
            add_noise=torch.normal(mean=mean, std=std, size=(200,)).reshape(1,2,-1).to(self.device)
            x=1/sqrt(self.alpha[t])*(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise

            # with open('D:/generate_2/airfoil_diffusion/generate.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     x_list = x.cpu().tolist()
            #     written_list = [t] + x_list#+[sqrtalpha]+[coeff_2.item()]+[coeff_3.item()]
            #     writer.writerow(written_list)
            #     print(f't={t}')
                    
        return x