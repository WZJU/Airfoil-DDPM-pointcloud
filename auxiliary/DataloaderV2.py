import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple, Union

class PointCloudACCDataset(Dataset):
    def __init__(self, pointcloud, context):
        """
        Args:
            pointcloud: NumPy数组 [B, N, C]
            context: NumPy数组 [B, 6]
        """
        self.pointcloud = torch.from_numpy(pointcloud).float()  # 转为Tensor [B, N, C]
        self.context = torch.from_numpy(context).float()  # 转为Tensor [B, 3]
        
        assert len(self.pointcloud) == len(self.context), "数据长度不一致"

    def __len__(self):
        return len(self.pointcloud)

    def __getitem__(self, idx):
        return {
            'pointcloud': self.pointcloud[idx],  # 形状 [N, C]
            'acc': self.context[idx],    # 形状 [6]
        }
    '''
    Dataloader使用示例
    for batch in dataloader:
        pointcloud_batch = batch['pointcloud']  # 形状 [batch_size, N, C]
        context_1 = batch['context_1']                # 形状 [batch_size, 3]
        context_2 = batch['context_2']                # 形状 [batch_size, 3]
    '''


class PointCloudAugmenter(nn.Module):
    def __init__(
        self,
        translate_range: Optional[Union[float, Tuple[float, float]]] = 0.1,
        scale_range: Optional[Union[float, Tuple[float, float]]] = (0.8, 1.2),
        per_dim_scale: bool = False,
        augment_prob: float = 0.5,
    ):
        """
        点云数据增强器（平移 + 缩放）
        
        Args:
            translate_range: 平移范围，可以是标量（各维度相同）或元组（各维度独立）
                            None 表示禁用平移。
            scale_range:     缩放范围，可以是标量（全局缩放）或元组（随机范围）
                            None 表示禁用缩放。
            per_dim_scale:   是否对每个特征维度独立缩放（True）或全局缩放（False）。
            augment_prob:    应用增强的概率（0~1）。
        """
        super().__init__()
        self.translate_range = translate_range
        self.scale_range = scale_range
        self.per_dim_scale = per_dim_scale
        self.augment_prob = augment_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, D, N] 的点云（D通常为3维坐标xyz）
        输出: 增强后的点云 [B, D, N]
        """
        if not self.training or self.augment_prob <= 0:
            return x

        B, D, N = x.shape
        device = x.device

        # 按概率决定是否增强
        if torch.rand(1, device=device) > self.augment_prob:
            return x

        # --------------------- 平移增强 ---------------------
        if self.translate_range is not None:
            if isinstance(self.translate_range, (float, int)):
                translate = torch.rand(B, D, 1, device=device) * 2 * self.translate_range - self.translate_range
            else:  # 各维度独立范围
                low, high = self.translate_range
                translate = torch.rand(B, D, 1, device=device) * (high - low) + low
            x = x + translate  # [B, D, N] + [B, D, 1]

        # --------------------- 缩放增强 ---------------------
        if self.scale_range is not None:
            if isinstance(self.scale_range, (float, int)):
                scale = torch.ones(B, D if self.per_dim_scale else 1, 1, device=device) * self.scale_range
            else:  # 随机范围缩放
                low, high = self.scale_range
                if self.per_dim_scale:
                    scale = torch.rand(B, D, 1, device=device) * (high - low) + low
                else:
                    scale = torch.rand(B, 1, 1, device=device) * (high - low) + low
            x = x * scale  # [B, D, N] * [B, D, 1] 或 [B, 1, 1]

        return x

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(translate_range={self.translate_range}, "
            f"scale_range={self.scale_range}, per_dim_scale={self.per_dim_scale}, "
            f"augment_prob={self.augment_prob})"
        )