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
from torch.nn import Module, Linear, ModuleList

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

class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret
    

class Unet(nn.Module):
    '''
    Input:  x:[B, num, dim]
            time_emb:[B, time_emb_dim(1)]
            context:[B, num(1), context_dim]
    Output: [B, num_out, dim_out]
    '''
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.silu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(point_dim, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, point_dim, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = [x]  # Initialize with input point cloud [B, N, D]
        length = len(self.layers)
        for i, layer in enumerate(self.layers):
            if i<= length//2:
                input_x = out[-1]
            else:
                input_x = out[-1] + out[length-i]
            output = layer(ctx=ctx_emb, x=input_x)
            if i < length - 1:
                out.append(self.act(output))
            else:
                out.append(output)

        if self.residual:
            return x + out[-1]
        else:
            return out[-1]
    
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

    def forward(self, x, context, CFG=1, t_max=500):
        mean = 0.5  # 均值
        std = 0.5  # 标准差
        Bs, n, _ = context.size()
        if x is None:
            x = torch.normal(mean=mean, std=std, size=(n*2,)).reshape(1,-1,2).to(self.device)
        else:
            xmean = x.mean()
            xstd = x.std()
            x = (x - xmean) / xstd*2

        # x[:, 1, :] *= 0.1

        x = x.repeat(Bs, 1, 1).to(self.device)
        for t in range(t_max-1, 0, -1):
            time_embedding = torch.tensor(t, dtype=torch.float32).unsqueeze(0).expand(Bs, -1).to(self.device)
            noise_pred=self.model(x, time_embedding, context=context)
            add_noise=torch.normal(mean=mean, std=std, size=(n*2,)).reshape(1,-1,2).to(self.device)
            x=1/sqrt(self.alpha[t])*(x-(1-self.alpha[t])*noise_pred/sqrt(1-self.alphabar[t]))+self.beta[t]*(1-self.alphabar[t-1])/(1-self.alphabar[t])*add_noise

            # with open('D:/generate_2/airfoil_diffusion/generate.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     x_list = x.cpu().tolist()
            #     written_list = [t] + x_list#+[sqrtalpha]+[coeff_2.item()]+[coeff_3.item()]
            #     writer.writerow(written_list)
            #     print(f't={t}')
                    
        return x
