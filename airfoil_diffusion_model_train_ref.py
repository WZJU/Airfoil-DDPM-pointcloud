import torch
from torch import nn
from torch.nn import functional as F
from airfoil_diffusion_model import Unet, airfoil_diffusion
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.tensorboard import SummaryWriter
#from mlp_data import MyData
import pandas as pd
#import h5py
import csv
from math import sqrt
import numpy as np

#Assistant fucntion
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

#定义存放位置
train_process='E:/wenzhe/generation2/airfoil_diffusion/train_3/step_info_c3.1.5m2.csv'
epoch_process='E:/wenzhe/generation2/airfoil_diffusion/train_3/epoch_info_c3.1.5m2.csv'
data_location='E:/wenzhe/generation2/airfoil_diffusion/train_3/ACcontext_combined_m2.csv'

#training Hyperparameters
batchsize = 150           # 每次加载的数据量
epochs = 100               # 数据变量次数
learningRate = 0.0005     # 初始化学习率
time_step=1000
result_model_mse = float("inf")  ##  定义训练结果模型是否保存的标志变量
result_model_mae = float("inf")
context_flag=True
CST_size=20
context_size=3
Total_size=CST_size+context_size#When context input
#定义model
df_model=Unet(1, 20,context_size_1=context_size, context_size_2=3, down_sampling_dim=2, dropout = 0.).cuda()# input_dim, query_size,context_size, down_sampling_dim,
checkpoint = torch.load('E:/wenzhe/generation2/airfoil_diffusion/models3/DFmodel_context_c3.1.5_100.pth', map_location=torch.device('cuda'),weights_only=True)   ### 加载神经网络模型
df_model = partial_load_state_dict(df_model, checkpoint)
epoch_0=100 #续算步数
#af=airfoil_diffusion(df_model)

loss = nn.MSELoss().cuda()
metric = nn.L1Loss().cuda()
writer = SummaryWriter("./mlp_logs/")
optim = torch.optim.Adam(df_model.parameters(), lr=learningRate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.3, last_epoch=-1)

##接下来定义数据处理部分


data = pd.read_csv(data_location)
data.fillna(0, inplace=True)
#手动进行归一化
data.iloc[:, 20] = (data.iloc[:, 20] - 0.007)/0.059
data.iloc[:, 21] = (data.iloc[:, 21] - 0.09)/0.14
data.iloc[:, 22] = (data.iloc[:, 22] - 0.33)/0.64
data.iloc[:, 23] = (data.iloc[:, 23] - 0)*10
data.iloc[:, 24] = np.sqrt(data.iloc[:, 24]) * 10
data.iloc[:, 25] = (data.iloc[:, 25] - 0)*10

total_length = len(data)
train_length = int(total_length * 0.75)
valid_length = int(total_length * 0.15)
test_length = total_length - train_length - valid_length

train_data = data.iloc[:train_length]
valid_data = data.iloc[train_length:train_length + valid_length]
test_data = data.iloc[train_length + valid_length:]


train_CST, train_context_1, train_context_2=train_data.iloc[:,:20].to_numpy(),train_data.iloc[:,20:23].to_numpy(),train_data.iloc[:,23:].to_numpy()
if train_context_1.size == 0:##For dataframe
    print('Data has No context1')
    context_flag_1=False
else:
    context_flag_1=True
if train_context_2.size == 0:##For dataframe
    print('Data has No context2')
    context_flag_2=False
else:
    context_flag_2=True
train_CST, train_context_1, train_context_2  = torch.from_numpy(train_CST).float(), torch.from_numpy(train_context_1).float(), torch.from_numpy(train_context_2).float()
train_data=TensorDataset(train_CST, train_context_1, train_context_2)
train_loader=DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True)

valid_CST, valid_context_1,valid_context_2=valid_data.iloc[:,:20].to_numpy(),valid_data.iloc[:,20:23].to_numpy(),valid_data.iloc[:,23:].to_numpy()
valid_CST, valid_context_1,valid_context_2  = torch.from_numpy(valid_CST).float(), torch.from_numpy(valid_context_1).float(), torch.from_numpy(valid_context_2).float()
valid_data=TensorDataset(valid_CST, valid_context_1,valid_context_2)
valid_loader=DataLoader(dataset=valid_data, batch_size=batchsize, shuffle=True)

test_CST, test_context_1, test_context_2=test_data.iloc[:,:20].to_numpy(),test_data.iloc[:,20:23].to_numpy(),test_data.iloc[:,23:].to_numpy()
test_CST, test_context_1, test_context_2  = torch.from_numpy(test_CST).float(), torch.from_numpy(test_context_1).float(), torch.from_numpy(test_context_2).float()
test_data=TensorDataset(test_CST, test_context_1, test_context_2)
test_loader=DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True)

#beta和alpha定义，用来混合噪声
def cal_alpha_bar(alpha, max_t):
    sqrtalphabar=torch.empty(max_t)
    sqrt_1_m_alphabar=torch.empty(max_t)
    alphabar_temp=1
    for i in range(max_t):
        alphabar_temp=alphabar_temp*alpha[i]
        sqrtalphabar[i]=sqrt(alphabar_temp)
        sqrt_1_m_alphabar[i]=sqrt(1-alphabar_temp)
    return sqrtalphabar,sqrt_1_m_alphabar

alpha=1-torch.linspace(0.0001, 0.02, steps=1000)
sqrtalphabar,sqrt_1_m_alphabar=cal_alpha_bar(alpha, 1000)

train_step=0
for epoch in range(epochs):
    print("---------------------开始第{}轮网络训练-----------------".format(epoch + 1 + epoch_0))
    train_loss = 0
    train_metric = 0
    train_number = 0    

    for data in train_loader:
        #处理正向扩散数据
        input, context_1, context_2 = data
        #print(input.size())
        #print(context.size())

        noise_tensor = torch.randn(time_step,input.size(0), 1, CST_size)
        input_tensor = input.unsqueeze(0).unsqueeze(2)
        input_tensor = input_tensor.expand(time_step, *input_tensor.shape[1:])#广播
        # 将一维张量扩展到与 input_tensor 和 noise_tensor 相匹配的维度
        sqrtalphabar_expanded = sqrtalphabar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(input_tensor)
        sqrt_1_m_alphabar_expanded = sqrt_1_m_alphabar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(noise_tensor)
        # 使用向量化操作进行计算
        input_tensor = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor

        #处理反向预测数据
        # 获取除了 t = 0 之外的切片
        noise_tensor_no_t0 = noise_tensor[1:]
        input_tensor_no_t0 = input_tensor[1:]
        # 沿着第一个维度展开
        noise_tensor_flattened = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0.shape[-2],noise_tensor_no_t0.shape[-1])
        input_tensor_flattened = input_tensor_no_t0.reshape(-1,input_tensor_no_t0.shape[-2],input_tensor_no_t0.shape[-1])
        # 获取 noise_tensor_flattened 的时间步总数
        N = noise_tensor_no_t0.shape[0]
        # 创建时间嵌入向量
        time_embedding = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(-1).expand(N, noise_tensor_no_t0.shape[1]) 
        time_embedding = time_embedding.reshape(N*noise_tensor_no_t0.shape[1],-1)

        time_embedding=time_embedding.cuda()
        input_tensor_flattened=input_tensor_flattened.cuda()
        noise_tensor_flattened=noise_tensor_flattened.cuda()

        if context_flag_1==True:
            context=context_1.unsqueeze(1).cuda()
            context=context.unsqueeze(0)
            #广播到时间维度上
            context = context.expand(time_step-1, *context.shape[1:])#广播,但是少一次，因为t=0的时候不用
            context = context.reshape(-1,context.shape[-2],context.shape[-1])

            #如果有context就训两次，一次带context，一次不带，先在else里面训一遍带context的
            noise_tensor_context = torch.randn(time_step,input.size(0), 1, CST_size)
            input_tensor_context = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor_context
            # 获取除了 t = 0 之外的切片
            noise_tensor_no_t0_context = noise_tensor_context[1:]
            input_tensor_no_t0_context = input_tensor_context[1:]
            # 沿着第一个维度展开
            noise_tensor_flattened_context = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0_context.shape[-2],noise_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context = input_tensor_no_t0.reshape(-1,input_tensor_no_t0_context.shape[-2],input_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context=input_tensor_flattened_context.cuda()
            noise_tensor_flattened_context=noise_tensor_flattened_context.cuda()
            pred_noise_context=df_model(input_tensor_flattened_context, time_embedding, context_1=context,context_2=None)
            step_loss=loss(pred_noise_context, noise_tensor_flattened_context)
            step_metric=metric(pred_noise_context, noise_tensor_flattened_context)
            optim.zero_grad()
            step_loss.backward()
            optim.step()
            #print(f'A batch with context has been completed! MSE={step_loss.item()} MAE={step_metric.item()}')
            train_step+=1
            current_lr = optim.param_groups[0]['lr']
            with open(train_process, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch +epoch_0+ 1,train_step,current_lr, step_loss.item(), step_metric.item()])

        if context_flag_2==True:
            context=context_2.unsqueeze(1).cuda()
            context=context.unsqueeze(0)
            #广播到时间维度上
            context = context.expand(time_step-1, *context.shape[1:])#广播,但是少一次，因为t=0的时候不用
            context = context.reshape(-1,context.shape[-2],context.shape[-1])

            #如果有context就训两次，一次带context，一次不带，先在else里面训一遍带context的
            noise_tensor_context = torch.randn(time_step,input.size(0), 1, CST_size)
            input_tensor_context = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor_context
            # 获取除了 t = 0 之外的切片
            noise_tensor_no_t0_context = noise_tensor_context[1:]
            input_tensor_no_t0_context = input_tensor_context[1:]
            # 沿着第一个维度展开
            noise_tensor_flattened_context = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0_context.shape[-2],noise_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context = input_tensor_no_t0.reshape(-1,input_tensor_no_t0_context.shape[-2],input_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context=input_tensor_flattened_context.cuda()
            noise_tensor_flattened_context=noise_tensor_flattened_context.cuda()
            pred_noise_context=df_model(input_tensor_flattened_context, time_embedding, context_2=context,context_1=None)
            step_loss=loss(pred_noise_context, noise_tensor_flattened_context)
            step_metric=metric(pred_noise_context, noise_tensor_flattened_context)
            optim.zero_grad()
            step_loss.backward()
            optim.step()
            #print(f'A batch with context has been completed! MSE={step_loss.item()} MAE={step_metric.item()}')
            train_step+=1
            current_lr = optim.param_groups[0]['lr']
            with open(train_process, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch +epoch_0+ 1,train_step,current_lr, step_loss.item(), step_metric.item()])

        if context_flag_1==True and context_flag_2==True:
            context_1=context_1.unsqueeze(1).cuda()
            context_1=context_1.unsqueeze(0)
            #广播到时间维度上
            context_1 = context_1.expand(time_step-1, *context_1.shape[1:])#广播,但是少一次，因为t=0的时候不用
            context_1 = context_1.reshape(-1,context_1.shape[-2],context_1.shape[-1])

            context_2=context_2.unsqueeze(1).cuda()
            context_2=context_2.unsqueeze(0)
            #广播到时间维度上
            context_2 = context_2.expand(time_step-1, *context_2.shape[1:])#广播,但是少一次，因为t=0的时候不用
            context_2 = context_2.reshape(-1,context_2.shape[-2],context_2.shape[-1])

            #如果有context就训两次，一次带context，一次不带，先在else里面训一遍带context的
            noise_tensor_context = torch.randn(time_step,input.size(0), 1, CST_size)
            input_tensor_context = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor_context
            # 获取除了 t = 0 之外的切片
            noise_tensor_no_t0_context = noise_tensor_context[1:]
            input_tensor_no_t0_context = input_tensor_context[1:]
            # 沿着第一个维度展开
            noise_tensor_flattened_context = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0_context.shape[-2],noise_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context = input_tensor_no_t0.reshape(-1,input_tensor_no_t0_context.shape[-2],input_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context=input_tensor_flattened_context.cuda()
            noise_tensor_flattened_context=noise_tensor_flattened_context.cuda()
            pred_noise_context=df_model(input_tensor_flattened_context, time_embedding, context_1=context_1,context_2=context_2)
            step_loss=loss(pred_noise_context, noise_tensor_flattened_context)
            step_metric=metric(pred_noise_context, noise_tensor_flattened_context)
            optim.zero_grad()
            step_loss.backward()
            optim.step()
            #print(f'A batch with context has been completed! MSE={step_loss.item()} MAE={step_metric.item()}')
            train_step+=1
            current_lr = optim.param_groups[0]['lr']
            with open(train_process, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch +epoch_0+ 1,train_step,current_lr, step_loss.item(), step_metric.item()])

        ####################################
        #开始无context训练
        pred_noise=df_model(input_tensor_flattened, time_embedding, context_1=None, context_2=None)
        step_loss=loss(pred_noise, noise_tensor_flattened)
        step_metric=metric(pred_noise, noise_tensor_flattened)
        optim.zero_grad()
        step_loss.backward()
        optim.step()
        train_number += 1  # 统计总计有多少个step
        train_loss += step_loss.item()  # 统计所有batchsize的总体损失
        train_metric+=step_metric.item()
        train_step+=1
        current_lr = optim.param_groups[0]['lr']
        with open(train_process, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch +epoch_0+ 1,train_step,current_lr, step_loss.item(), step_metric.item()])
        #Here, a batch of training has been completed.
        #print(f'A batch has been completed! MSE={step_loss.item()}')
        #Wow, looks like there is nothing to be done after a batch.
    #Here, an epoch has been completed.
    lr_scheduler.step()


    # #每几步需要对生成结果进行训练
    # if context_flag==True and (epoch+epoch_0)>=35:
    #     input_af=torch.empty(0).cuda()
    #     total_context=torch.empty(0).cuda()
    #     count=0

    #     for data in train_loader:
    #         input, context = data

    #         #对context进行处理
    #         context=context.unsqueeze(1).cuda()
    #         input=input.unsqueeze(1).cuda()

    #         input_af = torch.cat((input_af, input), dim=0)
    #         total_context = torch.cat((total_context, context), dim=0)
    #         count+=1
    #         #print(f'input_af size:{input_af.size()}, total_context size:{total_context.size()}')

    #         if count%5==0:
    #             #每5个batch就回传一次防止爆显存
    #             #这样只需要每5个batch生成一次
    #             output_airfoil=af(None, context=total_context, CFG=1, t_max=1000)

    #             generate_loss=loss(output_airfoil,input_af)
    #             target_loss_bw=generate_loss*0.001
    #             generate_metric=metric(output_airfoil,input_af)
    #             optim.zero_grad()
    #             if generate_loss>=1:
    #                 target_loss_bw=target_loss_bw*0.0000001
    #             target_loss_bw.backward()
    #             optim.step()
    #             print(f'5 Batches ,Generate Loss = {generate_loss}, Generate MAE = {generate_metric}')
    #             input_af=torch.empty(0).cuda()
    #             total_context=torch.empty(0).cuda()
    #             with open('E:/wenzhe/generation2/airfoil_diffusion/generation_evaluation_metrics_3.0.csv', 'a', newline='') as csvfile:
    #                 writer = csv.writer(csvfile)
    #                 writer.writerow([epoch + 1 + epoch_0,count/5,generate_loss.item(),generate_metric.item()])
            
        #Here, a batch of training has been completed.
        
        #Wow, looks like there is nothing to be done after a batch.


    #Validation Set, Start!
    valid_total_loss = 0
    valid_total_metric=0
    valid_number = 0
    valid_total_loss_context=0
    valid_total_metric_context=0
    valid_context_number=0
    for data in valid_loader:
        input, context_1, context_2 = data

        #统一处理的部分
        noise_tensor = torch.randn(time_step,input.size(0), 1, CST_size)
        input_tensor = input.unsqueeze(0).unsqueeze(2)
        input_tensor = input_tensor.expand(time_step, *input_tensor.shape[1:])#广播
        # 将一维张量扩展到与 input_tensor 和 noise_tensor 相匹配的维度
        sqrtalphabar_expanded = sqrtalphabar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(input_tensor)
        sqrt_1_m_alphabar_expanded = sqrt_1_m_alphabar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(noise_tensor)
        # 使用向量化操作进行计算
        input_tensor = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor

        #处理反向预测数据
        # 获取除了 t = 0 之外的切片
        noise_tensor_no_t0 = noise_tensor[1:]
        input_tensor_no_t0 = input_tensor[1:]
        # 沿着第一个维度展开
        noise_tensor_flattened = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0.shape[-2],noise_tensor_no_t0.shape[-1])
        input_tensor_flattened = input_tensor_no_t0.reshape(-1,input_tensor_no_t0.shape[-2],input_tensor_no_t0.shape[-1])
        # 获取 noise_tensor_flattened 的时间步总数
        N = noise_tensor_no_t0.shape[0]
        # 创建时间嵌入向量
        time_embedding = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(-1).expand(N, noise_tensor_no_t0.shape[1]) 
        time_embedding = time_embedding.reshape(N*noise_tensor_no_t0.shape[1],-1)

        time_embedding=time_embedding.cuda()
        input_tensor_flattened=input_tensor_flattened.cuda()
        noise_tensor_flattened=noise_tensor_flattened.cuda()


        if context_flag_1==True:
            context=context_1.unsqueeze(1).cuda()
            context=context.unsqueeze(0)
            #广播到时间维度上
            context = context.expand(time_step-1, *context.shape[1:])#广播,但是少一次，因为t=0的时候不用
            context = context.reshape(-1,context.shape[-2],context.shape[-1])

            #如果有context就训两次，一次带context，一次不带，先在else里面训一遍带context的
            noise_tensor_context = torch.randn(time_step,input.size(0), 1, CST_size)
            input_tensor_context = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor_context
            # 获取除了 t = 0 之外的切片
            noise_tensor_no_t0_context = noise_tensor_context[1:]
            input_tensor_no_t0_context = input_tensor_context[1:]
            # 沿着第一个维度展开
            noise_tensor_flattened_context = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0_context.shape[-2],noise_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context = input_tensor_no_t0.reshape(-1,input_tensor_no_t0_context.shape[-2],input_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context=input_tensor_flattened_context.cuda()
            noise_tensor_flattened_context=noise_tensor_flattened_context.cuda()
            with torch.no_grad():
                pred_noise_context=df_model(input_tensor_flattened_context, time_embedding, context_1=context, context_2=None)
                valid_loss=loss(pred_noise_context, noise_tensor_flattened_context)
                valid_metric=metric(pred_noise_context, noise_tensor_flattened_context)
                valid_total_loss_context+=valid_loss.item()
                valid_total_metric_context+=valid_metric.item()
                valid_context_number+=1
        
        if context_flag_2==True:
            context=context_2.unsqueeze(1).cuda()
            context=context.unsqueeze(0)
            #广播到时间维度上
            context = context.expand(time_step-1, *context.shape[1:])#广播,但是少一次，因为t=0的时候不用
            context = context.reshape(-1,context.shape[-2],context.shape[-1])

            #如果有context就训两次，一次带context，一次不带，先在else里面训一遍带context的
            noise_tensor_context = torch.randn(time_step,input.size(0), 1, CST_size)
            input_tensor_context = sqrtalphabar_expanded * input_tensor + sqrt_1_m_alphabar_expanded * noise_tensor_context
            # 获取除了 t = 0 之外的切片
            noise_tensor_no_t0_context = noise_tensor_context[1:]
            input_tensor_no_t0_context = input_tensor_context[1:]
            # 沿着第一个维度展开
            noise_tensor_flattened_context = noise_tensor_no_t0.reshape(-1,noise_tensor_no_t0_context.shape[-2],noise_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context = input_tensor_no_t0.reshape(-1,input_tensor_no_t0_context.shape[-2],input_tensor_no_t0_context.shape[-1])
            input_tensor_flattened_context=input_tensor_flattened_context.cuda()
            noise_tensor_flattened_context=noise_tensor_flattened_context.cuda()
            with torch.no_grad():
                pred_noise_context=df_model(input_tensor_flattened_context, time_embedding, context_1=None,context_2=context)
                valid_loss=loss(pred_noise_context, noise_tensor_flattened_context)
                valid_metric=metric(pred_noise_context, noise_tensor_flattened_context)
                valid_total_loss_context+=valid_loss.item()
                valid_total_metric_context+=valid_metric.item()
                valid_context_number+=1


        #开始无context验证
        with torch.no_grad():
            pred_noise=df_model(input_tensor_flattened, time_embedding,  context_1=None, context_2=None)
            valid_loss=loss(pred_noise, noise_tensor_flattened)
            valid_metric=metric(pred_noise, noise_tensor_flattened)
        valid_number += 1  # 统计总计有多少个step
        valid_total_loss += valid_loss.item()  # 统计所有batchsize的总体损失
        valid_total_metric+=valid_metric.item()
        #Here, a batch of validation has been completed.
        #print('A batch has been completed!')
        #Wow, looks like there is nothing to be done after a batch.

    avg_train_loss=train_loss/train_number
    avg_train_metric=train_metric/train_number
    avg_valid_loss=valid_total_loss/valid_number
    avg_valid_metric=valid_total_metric/valid_number
    avg_valid_loss_context=valid_total_loss_context/valid_context_number
    avg_valid_metric_context=valid_total_metric_context/valid_context_number
    #Boawh, every batch of valid data has been completed, let us look at them
    print("Epoch:{},训练集上的Loss：{}, MAE:{};验证数据集上的Loss:{}, MAE:{}; 考虑context, Loss= {}, MAE= {}".format(epoch + 1 + epoch_0, avg_train_loss, avg_train_metric, avg_valid_loss, avg_valid_metric,avg_valid_loss_context,avg_valid_metric_context))
    #写入计算数据
    with open(epoch_process, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1 + epoch_0, avg_train_loss, avg_valid_loss, avg_train_metric, avg_valid_metric,avg_valid_loss_context,avg_valid_metric_context])
    model_state = {'models': df_model.state_dict(),      ### 模型保存用于神经网络模型的加载与续算
                     'optimizer': optim.state_dict(),
                     'epoch': epoch + 1 + epoch_0}
    
    torch.save(model_state, "E:/wenzhe/generation2/airfoil_diffusion/models3/DFmodel_context_c3.1.5m2_{}.pth".format(epoch + 1 +epoch_0))
    print("模型已保存！")
        

