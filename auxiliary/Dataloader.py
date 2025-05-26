import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudACCDataset(Dataset):
    def __init__(self, pointcloud, context):
        """
        Args:
            pointcloud: NumPy数组 [B, N, C]
            context: NumPy数组 [B, 6]
        """
        self.pointcloud = torch.from_numpy(pointcloud).float()  # 转为Tensor [B, N, C]
        self.context_1 = torch.from_numpy(context[:,:3]).float()  # 转为Tensor [B, 3]
        self.context_2 = torch.from_numpy(context[:,3:]).float()  # 转为Tensor [B, 3]
        
        assert len(self.pointcloud) == len(self.context_1), "数据长度不一致"

    def __len__(self):
        return len(self.pointcloud)

    def __getitem__(self, idx):
        return {
            'pointcloud': self.pointcloud[idx],  # 形状 [N, C]
            'context_1': self.context_1[idx],    # 形状 [3]
            'context_2': self.context_2[idx]     # 形状 [3]
        }
    '''
    Dataloader使用示例
    for batch in dataloader:
        pointcloud_batch = batch['pointcloud']  # 形状 [batch_size, N, C]
        context_1 = batch['context_1']                # 形状 [batch_size, 3]
        context_2 = batch['context_2']                # 形状 [batch_size, 3]
    '''
