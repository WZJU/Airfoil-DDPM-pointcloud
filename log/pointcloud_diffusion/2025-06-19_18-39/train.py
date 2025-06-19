# This file is training code of Airfoil_DDPM.
# Author: Zhe Wen
# Date: 2025-5-22
# Copyright (c) Zhejiang University. All rights reserved.
# See LICENSE file in the project root for license information.

import torch
from torch import nn
from torch.nn import functional as F
from Airfoil_DDPM_pointcloudV3 import Unet
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
#from mlp_data import MyData
import pandas as pd
#import h5py
import csv
from math import sqrt
import numpy as np


import auxiliary.argument_parser as argument_parser
import auxiliary.DataloaderV2 as Dataloader
import datetime
import logging
from pathlib import Path
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def generate_gaussian_tensor(batch_size, dim, size):
    tensor = torch.empty(batch_size, dim, size)
    for i in range(batch_size):
        for j in range(dim):
            tensor[i, j] = torch.normal(mean=0.0, std=1.0, size=(size,))
    return tensor

def partial_load_state_dict(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = checkpoint['models']
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    return model

'''beta和alpha定义，用来混噪声合'''
def cal_alpha_bar(alpha, max_t):
    '''
    alpha : torch[max_t]
    sqrtalphabar, sqrt_1_m_alphabar : torch[max_t]
    '''
    sqrtalphabar=torch.empty(max_t)
    sqrt_1_m_alphabar=torch.empty(max_t)
    alphabar_temp=1
    for i in range(max_t):
        alphabar_temp=alphabar_temp*alpha[i]
        sqrtalphabar[i]=sqrt(alphabar_temp)
        sqrt_1_m_alphabar[i]=sqrt(1-alphabar_temp)
    return sqrtalphabar, sqrt_1_m_alphabar

def random_sample_points(input, num_samples=100):
    B, D, N = input.shape
    # 生成随机索引（每个样本独立采样，不重复）
    indices = torch.stack([torch.randperm(N)[:num_samples] for _ in range(B)], dim=0)  # [B, 100]
    indices = indices.unsqueeze(1).expand(-1, D, -1)  # [B, D, 100]
    # 使用 gather 采样
    sampled = torch.gather(input, dim=2, index=indices.to(input.device))
    return sampled

def uniform_sample_t(num_steps,batch_size):
    ts = np.random.choice(np.arange(1, num_steps), batch_size)
    ts_tensor = torch.tensor(ts)
    return ts_tensor

def forward_propagation(model, loss, metric, input, context=None, time_step = 500, device = 'cuda', data_aug=None):
    '''
    Input:
        input: Tensor [B, N, D]
        context : Tensor [B, 1, D(3)]
    return: loss, metric
    '''
    with torch.no_grad():
        batch_size, _, point_dim = input.size()
        time_embedding = uniform_sample_t(time_step, batch_size)
        input = random_sample_points(input, num_samples=100)#[B, D, N sample]

        c0 = sqrtalphabar[time_embedding].view(-1, 1, 1)        # (B, 1, 1)
        c1 = sqrt_1_m_alphabar[time_embedding].view(-1, 1, 1) 

        e_rand = torch.randn_like(input)  # (B, N, d)
        noise_input = c0 * input + c1 * e_rand

        noise_input = noise_input.to(device)
        time_embedding = time_embedding.to(device)
        context = context.to(device)
        e_rand = e_rand.to(device)

    e_theta = model(noise_input, time_embedding, context)
    loss = loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim))
    metric = metric(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim))
    return loss, metric

def save_info(epoch, current_lr, loss, log_dir, type='step'):
    if type=='step':
        train_process=os.path.join(log_dir, 'step_info.csv')
    elif type=='epoch':
        train_process=os.path.join(log_dir, 'epoch_info.csv')

    with open(train_process, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, current_lr, loss])

def main(opt):
    def log_string(str):
        logger.info(str)
        print(str)
    
    '''DEVICE'''
    if opt.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        if opt.device == 'cuda':  # 用户想用 GPU 但不可用时警告
            print("[Warning] CUDA not available. Falling back to CPU.")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path(os.path.join('log'))
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(opt.model)
    exp_dir.mkdir(exist_ok=True)
    if opt.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(opt.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    #args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, opt.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(opt)

    train_process = os.path.join(log_dir, "train_process.csv")
    valid_process = os.path.join(log_dir, "valid_process.csv")

    #copy training code and parser
    shutil.copy(os.path.join('train.py'), str(exp_dir))
    shutil.copy(os.path.join('auxiliary/argument_parser.py'), str(exp_dir))
    shutil.copy(os.path.join('Airfoil_DDPM_pointcloud.py'), str(exp_dir))

    '''Init'''
    df_model=Unet(point_dim=2, context_dim=6, residual=False).to(device)#
    # checkpoint = torch.load('E:/wenzhe/generation2/airfoil_diffusion/models3/DFmodel_context_c3.1.5_100.pth', map_location=torch.device(device),weights_only=True)   ### 加载神经网络模型
    # df_model = partial_load_state_dict(df_model, checkpoint)

    loss = nn.MSELoss().to(device)
    metric = nn.L1Loss().to(device)
    optim = torch.optim.Adam(df_model.parameters(), lr=opt.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.3, last_epoch=-1)

    time_step = opt.time_step
    alpha=1-torch.linspace(0.0001, 0.02, steps=time_step)
    global sqrtalphabar,sqrt_1_m_alphabar
    sqrtalphabar,sqrt_1_m_alphabar = cal_alpha_bar(alpha, time_step)

    '''Load Data'''
    if opt.data_path is not None:
        data_path = opt.data_path
    else:
        data_path = os.path.join("../data","train_data_pointcloud.npz")
    data = np.load(data_path)
    # loaded_pointcloud = np.transpose(data['pointcloud'],(0,2,1))  # 形状 [B, N, D]-> [B, D, N]
    loaded_pointcloud = np.transpose(data['pointcloud'],(0,1,2))    #[B, N, D]
    loaded_pointcloud[:, 1, :] *= 10
    loaded_ACC = data['ACC']                # 形状 [B, 6]

    '''Dataloader'''
    # 划分数据集，创建Dataset
    len = loaded_pointcloud.shape[0]
    train_len = int(len*opt.train_split)
    valid_len = int(len*opt.valid_split)
    train_dataset = Dataloader.PointCloudACCDataset(loaded_pointcloud[:train_len], loaded_ACC[:train_len])
    valid_dataset = Dataloader.PointCloudACCDataset(loaded_pointcloud[train_len:valid_len+train_len], loaded_ACC[train_len:valid_len+train_len])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,#这不用注释了吧
        shuffle=opt.shuffle,# 打乱
        num_workers=opt.num_workers,# 多进程加载数
        pin_memory=True  # 加速GPU传输
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,#这不用注释了吧
        shuffle=opt.shuffle,# 打乱
        num_workers=opt.num_workers,# 多进程加载数
        pin_memory=True  # 加速GPU传输
    )

    '''
    Dataloader使用示例
    for batch in dataloader:
        pointcloud_batch = batch['pointcloud']  # 形状 [batch_size, N, D]
        acc_batch = batch['acc']                # 形状 [batch_size, 6]
    '''
    '''
    设置数据增强
    '''
    augmenter = Dataloader.PointCloudAugmenter(
        translate_range=(-0.3, 0.3),  # 各维度在[-0.1, 0.1]内随机平移
        scale_range=(0.9, 1.1),       # 全局缩放范围[0.9, 1.1]
        per_dim_scale=False,          # 全局缩放
        augment_prob=0.8,             # 80%概率增强
    )

    train_step=0
    best_loss = 1e6
    for epoch in range(opt.epochs):
        log_string("---------------------开始第{}轮网络训练-----------------".format(epoch+1))

        train_loss = 0
        train_metric = 0
        train_number = 0

        for batch in train_dataloader:
            pointcloud_batch = batch['pointcloud']  # 形状 [batch_size, D, N]
            # context_1 = batch['context_1']                # 形状 [batch_size, 3]
            # context_2 = batch['context_2']                # 形状 [batch_size, 3]
            context = batch['acc']# 形状 [batch_size, 6]

            #第一次正向传播和反向传播
            step_loss, step_metric = forward_propagation(df_model, loss, metric, pointcloud_batch, context, time_step = time_step, device = opt.device, data_aug=augmenter)
            optim.zero_grad()
            step_loss.backward()
            optim.step()
            train_step += 1
            train_number += 1
            train_loss += step_loss.item()
            train_metric += step_metric.item()
            current_lr = optim.param_groups[0]['lr']
            with open(train_process, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1,train_step,current_lr, step_loss.item(), step_metric.item()])
            
            
        lr_scheduler.step()
        torch.cuda.empty_cache()
        log_string(f"[TRAIN] Epoch: {epoch+1}, MSE = {train_loss/train_number}, MAE = {train_metric/train_number}")
                    
        
        valid_loss = 0
        valid_metric = 0
        valid_number = 0
        for batch in valid_dataloader:
            pointcloud_batch = batch['pointcloud']  # 形状 [batch_size, D, N]
            # context_1 = batch['context_1']                # 形状 [batch_size, 3]
            # context_2 = batch['context_2']                # 形状 [batch_size, 3]
            context = batch['acc']# 形状 [batch_size, 6]

            #正向传播
            with torch.no_grad():
                step_loss, step_metric = forward_propagation(df_model.eval(), loss, metric, pointcloud_batch, context, time_step = time_step, device = opt.device)
                valid_number += 1
                valid_loss += step_loss.item()
                valid_metric +=step_metric.item()
            with open(valid_process, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch + 1, step_loss.item(), step_metric.item()])
                    
        torch.cuda.empty_cache()  
        log_string(f"[VALID] Epoch: {epoch+1}, MSE = {valid_loss/valid_number}, MAE = {valid_metric/valid_number}")


        '''保存模型'''
        if  valid_loss/valid_number < best_loss:
            best_loss = valid_loss/valid_number
            model_state = {'models': df_model.state_dict(),      ### 模型保存用于神经网络模型的加载与续算
                    'optimizer': optim.state_dict(),
                    'epoch': epoch + 1}
            best_model_path = checkpoints_dir / 'best_model.pth'  # 拼接路径 
            torch.save(model_state, best_model_path)  # 保存模型权重


if __name__ == '__main__':
    opt = argument_parser.parser()
    main(opt)



