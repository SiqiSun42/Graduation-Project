# import torch
#
# from model import generate_model  # 使用生成模型的函数
#
# class DeepSVDD(object):
#     """核心的 Deep SVDD 类，只计算损失"""
#
#     def __init__(self, objective: str = 'one-class', nu: float = 0.1,  args=None):
#         """初始化 DeepSVDD 模型"""
#         assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
#         self.objective = objective
#         assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
#         self.nu = nu
#         self.R = 0.0  # 超球体半径 R
#         self.c = None  # 超球体中心 c
#         self.results = {
#             'train_time': None,
#             'test_auc': None,
#             'test_time': None,
#             'test_scores': None,
#         }
#         self.model = generate_model(args)  # 使用 generate_model 函数加载模型
#         #print("self.model:", self.model)  # 打印模型信息
#
#
#     def update_hypersphere_params(self, normed_vec):
#         """
#         更新超球体的中心 c 和半径 R。
#         这里的 R 是通过训练样本到中心的距离计算得到的。
#         """
#         # 计算超球体中心 c（所有正常样本的特征均值）
#         self.c = torch.mean(normed_vec, dim=0)  # 求所有样本的特征均值
#
#         # 计算每个样本到中心的距离
#         distances = torch.norm(normed_vec - self.c, p=2, dim=1)
#
#         # 使用距离的中位数计算半径 R，以避免极端值
#         self.R = torch.median(distances).item()  # 使用中位数来避免过大或过小的半径
#         # 或者可以使用均值来计算半径：self.R = torch.mean(distances).item()
#
#     def compute_loss(self, normed_vec):
#         """
#         计算与超球体的距离损失，使用 L2 距离（欧式距离）来计算每个样本的损失。
#         normed_vec: 输入数据，已经归一化的特征向量
#         """
#         # 如果超球体中心 c 尚未初始化，则在此使用 normed_vec 更新超球体参数
#         if self.c is None:
#             self.update_hypersphere_params(normed_vec)
#
#         # 确保 self.c 和 normed_vec 的形状匹配
#         if self.c.dim() == 1 and self.c.shape[0] == normed_vec.shape[1]:
#             self.c = self.c.unsqueeze(0).expand(normed_vec.shape[0], -1)
#
#         # 计算每个样本到超球体中心的欧式距离
#         distance = torch.norm(normed_vec - self.c, p=2, dim=1)
#
#         # 计算损失
#         if self.objective == 'one-class':
#             # "one-class" 设置下的损失：正常样本与超球体中心的距离
#             loss = torch.mean(distance)  # 计算所有样本到超球体中心的平均距离
#
#         elif self.objective == 'soft-boundary':
#             # "soft-boundary" 设置下的损失：允许一些样本接近边界，但仍然计算距离
#             scores = distance - self.R  # 计算每个样本与超球体半径的偏差
#             loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))  # 最大值与0比较，确保小于半径的样本不会增加损失
#
#         else:
#             raise ValueError("Objective must be either 'one-class' or 'soft-boundary'")
#
#         return loss
#
#
#
#
#
import json
import torch

from model import generate_model  # 使用生成模型的函数

class DeepSVDD(object):
    """核心的 Deep SVDD 类，只计算损失"""

    def __init__(self, objective: str = 'one-class', nu: float = 0.1,  args=None):
        """初始化 DeepSVDD 模型"""
        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # 超球体半径 R
        self.c = None  # 超球体中心 c
        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }
        self.model = generate_model(args)  # 使用 generate_model 函数加载模型
        #print("self.model:", self.model)  # 打印模型信息

    def update_hypersphere_params(self, normed_vec):
        """
        使用正常样本的归一化特征向量来更新超球体的中心和半径。
        data: 输入数据，用于从模型中提取特征。
        """
        # 更新超球体的中心 c 为正常样本的均值
        self.c = torch.mean(normed_vec, dim=0, keepdim=False)  # 将所有正常样本的特征向量取均值作为中心

        # 使用样本到中心的最大距离作为半径 R
        distances = torch.norm(normed_vec - self.c, p=2, dim=1)

        # 修改：使用距离的中位数或均值来设置半径，而不是最大值
        self.R = torch.median(distances).item()  # 使用中位数计算半径，避免过大
        # 或者你也可以使用均值计算半径：self.R = torch.mean(distances).item()

    def compute_loss(self, normed_vec):
        """
        计算与超球体的距离损失。
        data: 输入数据，用于从模型中提取特征。
        """
        #print(f'normed_vec.shape: {normed_vec.shape}')  # 打印 normed_vec 的形状
        #print(f'self.c.shape: {self.c.shape}')  # 打印 self.c 的形状

        if self.c is None:
            raise ValueError("Hyper-sphere center `c` has not been initialized.")

        # 确保 self.c 是与 normed_vec 的形状匹配
        # 如果 self.c 是一个 1D 向量（长度为 512），将其扩展为与 normed_vec 相同的形状
        if self.c.dim() == 1 and self.c.shape[0] == normed_vec.shape[1]:
            self.c = self.c.unsqueeze(0).expand(normed_vec.shape[0], -1)  # 扩展成 [batch_size, feature_dim]

        if self.c.shape[0] != normed_vec.shape[0]:
            print("Shape mismatch: Returning loss = -1.")
            return -1  # 返回 loss = -1 并结束函数

        # 计算每个样本到超球体中心的欧氏距离
        distance = torch.norm(normed_vec - self.c, p=2, dim=1)

        # 对于 "one-class" 设置，超球体外的点才有损失
        if self.objective == 'one-class':
            loss = torch.mean(torch.clamp(distance - self.R, min=-1))  # 超过半径 R 的部分作为损失

        # 对于 "soft-boundary" 设置，允许有些点在边界附近
        elif self.objective == 'soft-boundary':
            loss = torch.mean(torch.relu(distance - self.R))  # 使用 ReLU 激活，以平滑损失

        else:
            raise ValueError("Objective must be either 'one-class' or 'soft-boundary'")

        return loss

    def save_results(self, export_json):
        """保存结果字典为 JSON 文件"""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
