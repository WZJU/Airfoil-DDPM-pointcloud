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

#对某一层做残差连接
class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
#downsampling and up sampling，实际上是降维和升维的过程
def Upsample(num_in, num_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(num_in, default(num_out, num_in), 3, padding = 1)#输入维度是dim，输出维度如果dim_out有值则为dim_out，否则为dim
    )

def Downsample(num_in, num_out = None):
    return nn.Conv1d(num_in, default(num_out, num_in), 4, 2, 1)#卷积核尺寸为4，stride=2，padding=1；由于stride=2，所以输出的尺寸刚好会是输入的一半

def Dim_modify(dim_in, dim_out = None):
    return nn.Sequential(
        nn.Conv1d(dim_in, default(dim_out, dim_in), kernel_size = 3, padding = 1)#输入维度是num_in，输出维度如果dim_out有值则为num_out，否则为num
    )

def Num_modify(Num_in, Num_out):
    return nn.Sequential(
        nn.Linear(Num_in, Num_out)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class Block(Module):#包含卷积、归一化、激活和 Dropout 等操作的函数
    ''' Input : [B, dim, num_in]
        Output : [B, dim, num_out]
    '''
    def __init__(self, dim, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(in_channels = dim, out_channels = dim, kernel_size = 3, padding = 1)#一维卷积
        self.norm = RMSNorm(dim)
        self.act = nn.SiLU()  #Sigmoid
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    '''
    Input :  [B, dim, num_in]
    Output:  [B, dim, num_out]
    '''
    def __init__(self, dim_in, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim_in, dropout = dropout)
        self.block2 = Block(dim_in)
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):#如果有time embedding，则用MLP对time embedding进行处理
            time_emb = time_emb / 500 
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)#用time_emb得到的scale_shift对x进行缩放

        h = self.block2(h)

        return h + self.res_conv(x)#用MLP作为f（x），用res_conv进行维度调整、残差连接

    
class CrossAttention(nn.Module):
    '''
    Input :  query [B, query_dim, num]
             context [B, num, context_dim]
    Output : [B, query_dim, num]
    '''
    def __init__(self, query_dim, out_dim, context_dim=None, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        context_dim = default(context_dim, query_dim)

        self.proj_q = nn.Sequential(
            nn.Linear(query_dim, dim_head)
        )
        self.proj_k = nn.Sequential(
            nn.Linear(query_dim, dim_head)   
        )
        self.proj_v = nn.Sequential(
            nn.Linear(query_dim, dim_head)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_head, out_dim),
            nn.Dropout(dropout)
        )

        if context_dim is not None:
            self.proj_c2k = nn.Sequential(
            nn.Linear(context_dim, 40, bias=False),
            nn.ReLU(),
            nn.Linear(40, dim_head, bias=False)
        )
        self.cross_attn = MultiheadAttention(
            embed_dim=dim_head,
            num_heads=heads,
            batch_first=True
        )

    def forward(self, x, context=None, mask=None):
        x = x.permute(0,2,1) #[B, query_dim, num] -> [B, num, query_dim]
        q = self.proj_q(x)
        v = self.proj_v(x)
        if context is not None:
            context=context.repeat(1, x.shape[1], 1)
            k = self.proj_c2k(context) # [B, H*W, inner_size]
        else:
            k = self.proj_k(x)


        attn_output, _ = self.cross_attn(
                    query=q,
                    key=k,
                    value=v
                )

        out = self.to_out(attn_output)
        return out.permute(0,2,1) #[B, num, query_dim]  -> [B, query_dim, num]

class PointCloudSelfAttention(nn.Module):
    """
    点云自注意力机制（可改变点数）
    
    输入:
        x: [B, dim, num_in]  输入点云
    输出:
        [B, dim, num_out]  输出点云
        
    参数:
        - dim: 点云特征维度
        - num_out: 输出点数
        - num_heads: 注意力头数
        - dim_head: 每个头的维度
        - dropout: dropout率
    """
    def __init__(self, dim, num_out, num_heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.num_out = num_out
        inner_dim = dim_head * num_heads
        
        # 可学习的查询点(决定输出点的位置和数量)
        self.query_pos = nn.Parameter(torch.randn(1, dim, num_out))
        
        # 投影层
        self.to_q = nn.Conv1d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(dim, inner_dim * 2, 1, bias=False)
        
        # 输出转换
        self.to_out = nn.Sequential(
            nn.Conv1d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5

    def forward(self, x):
        """
        x: [B, dim, num_in] 输入点云
        """
        B, dim, num_in = x.shape
        
        # 生成查询 - 使用可学习的位置编码
        queries = repeat(self.query_pos, '1 d n -> b d n', b=B)
        
        # 投影到Q, K, V
        q = self.to_q(queries)  # [B, inner_dim, num_out]
        k, v = self.to_kv(x).chunk(2, dim=1)  # [B, inner_dim, num_in] 和 [B, inner_dim, num_in]
        
        # 分割多头
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=self.num_heads), (q, k, v))
        
        # 计算注意力
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        # 应用注意力
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        
        return self.to_out(out)  # [B, dim, num_out]

    
class UNet_Block_down(nn.Module):
    '''
    Input:  [B, query_dim_in, num_in]
            [B, time_emb_dim]
    Output: [B, query_dim_out, num_out]
    '''
    def __init__(self, num_in, num_out, dim_in, dim_out, context_dim, time_emb_dim=None, dropout = 0.):
        super().__init__()
        self.resnet=ResnetBlock(dim_in, dim_in, time_emb_dim = time_emb_dim, dropout = dropout)
        self.nummodify = Num_modify(num_in, num_out)
        self.crossattention=CrossAttention(query_dim=dim_in, out_dim=dim_out, context_dim=context_dim, heads=4, dim_head=32, dropout=dropout)
        

    def forward(self, x,time_emb = None, context=None, num_mod_first = True):
        x=self.resnet(x,time_emb)
        x=self.nummodify(x)
        x=self.crossattention(x, context)
        return x


class UNet_Block_up(nn.Module):
    '''
    Input:  [B, query_dim_in, num_in]
            [B, time_emb_dim]
    Output: [B, query_dim_out, num_out]
    '''
    def __init__(self, num_in, num_out, dim_in, dim_out, context_dim, time_emb_dim=None, dropout = 0.):
        super().__init__()
        self.resnet=ResnetBlock(dim_in, dim_in, time_emb_dim = time_emb_dim, dropout = dropout)
        self.crossattention=CrossAttention(query_dim=dim_in, out_dim=dim_out, context_dim=context_dim, heads=4, dim_head=32, dropout=dropout)
        self.cross_attn = PointCloudSelfAttention(dim = dim_out, num_out = num_out, num_heads=8, dim_head=64,)
        

    def forward(self, x,time_emb = None, context=None, num_mod_first = True):
        x=self.resnet(x,time_emb)
        x=self.crossattention(x, context)
        x=self.cross_attn(x)
        return x
    
    

class Unet(nn.Module):
    '''
    Input:  x:[B, dim, num]
            time_emb:[B, time_emb_dim(1)]
            context:[B, num(1), context_dim]
    Output: [B, dim_out, num_out]
    '''
    def __init__(self, num = [200, 10, 200], dim = [2, 2, 2], context_dim_1 = 3, context_dim_2 = 3, dropout = 0.):
        super().__init__()
        self.block_down_1=UNet_Block_down( num_in = num[0], num_out = num[1], dim_in = dim[0], dim_out = dim[1], context_dim = context_dim_1, time_emb_dim = 1, dropout = dropout)
        self.block_up_1=UNet_Block_up( num_in = num[1], num_out = num[2], dim_in = dim[1], dim_out = dim[2], context_dim = context_dim_2, time_emb_dim = 1, dropout = dropout)

    def forward(self, x, time_emb, context_1=None, context_2=None):
        d1=self.block_down_1( x, time_emb, context = context_1)
        u1=self.block_up_1( d1, context = context_2)
        return u1
    
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
    def __init__(self, model):
        super().__init__()
        #self.model=Unet(input_dim, query_size,context_size, down_sampling_dim, dropout = 0.)
        # 模型加载
        #checkpoint = torch.load(model_filename, map_location=torch.device('cuda'),weights_only=True)   ### 加载神经网络模型
        #self.model.load_state_dict(checkpoint['models'])
        self.model=model
        on_cuda = next(self.model.parameters()).is_cuda

        self.beta=torch.linspace(0.0001, 0.02, steps=1000)
        self.alpha=1-self.beta
        self.alphabar=cal_alpha_bar(self.alpha, 1000)
        # 如果模型在 CUDA 上，将张量也移动到 CUDA
        if on_cuda:
            self.beta = self.beta.cuda()
            self.alpha = self.alpha.cuda()
            self.alphabar = self.alphabar.cuda()
            self.model = self.model.cuda()

    def forward(self, x, context_1, context_2, CFG=1, t_max=1000):
        mean = 0.0  # 均值
        std = 1.0  # 标准差
        if x is None:
            x = torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
            if context_1 is not None:
                 shape_1=context_1.shape[0]
            else:
                shape_1=1
            if context_2 is not None:
                 shape_2=context_2.shape[0]
            else:
                shape_2=1

            x.repeat(shape_1*shape_2, 1, 1).cuda()
        else:
            xmean = x.mean()
            xstd = x.std()
            x = (x - xmean) / xstd*2
            x=x.repeat(context_1.shape[0]*context_2.shape[0], 1, 1)
        context_1_combined,context_2_combined=tensor_combine(context_1,context_2)
        for t in range(t_max-1, 0, -1):
            time_embedding = torch.tensor(t, dtype=torch.float32).unsqueeze(0).expand(x.shape[0], -1).cuda()
            hidden_without_context=self.model(x, time_embedding, context_1=None, context_2=None)
            hidden_with_context=self.model(x, time_embedding, context_1=context_1_combined, context_2=context_2_combined)
            #因为是多目标，所以需要对hidden_with_context进行处理
            hidden_with_context=weighted_average(hidden_with_context)
            noise_pred= hidden_without_context + CFG*( hidden_with_context-hidden_without_context)
            add_noise=torch.normal(mean=mean, std=std, size=(20,)).reshape(1,-1,20).cuda()
            x=1/sqrt(self.alpha[t])*(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise

            # with open('D:/generate_2/airfoil_diffusion/generate.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     x_list = x.cpu().tolist()
            #     written_list = [t] + x_list#+[sqrtalpha]+[coeff_2.item()]+[coeff_3.item()]
            #     writer.writerow(written_list)
            #     print(f't={t}')
                    
        return x