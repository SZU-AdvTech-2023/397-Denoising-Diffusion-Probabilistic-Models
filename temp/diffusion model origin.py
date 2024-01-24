"""
@Author: Alex
@Date: 2022.Dec.6th
Diffusion model
"""
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import torch.nn as nn
import io
from PIL import Image

num_steps = 100
# 制定每一步的beta，beta按照时间从小到大变化
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas
# alpha连乘
alphas_prod = torch.cumprod(alphas, 0)
# 从第一项开始，第0项另乘1？？？
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
# alphas_prod开根号
alphas_bar_sqrt = torch.sqrt(alphas_prod)
# 之后公式中要用的
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
# 大小都一样，常数不需要训练
assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
       alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
       == one_minus_alphas_bar_sqrt.shape
print("all the same shape", betas.shape)
#定义model输入维度
in_feature = 2000

class CustomDataset(Dataset):
    def __init__(self, csv_file, drop_column=None, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.drop_column = drop_column
        if drop_column:
            self.data_frame = self.data_frame.drop(columns=[drop_column])
        self.transform = transform


    def __len__(self):
        return len(self.data_frame.columns)

    def __getitem__(self, idx):
        feature = self.data_frame.iloc[:, idx].values
        sample = {'feature': feature}

        if self.transform:
            sample = self.transform(sample)

        return sample

# 定义一个简单的转换，将 NumPy 数组转换为 PyTorch 张量
class ToTensor:
    def __call__(self, sample):
        feature = sample['feature']
        return {'feature': torch.tensor(feature, dtype=torch.float32)}


#给定初始，算出任意时刻采样值——正向扩散
# 计算任意时刻的x采样值，基于x_0和重参数化
def q_x(x_0, t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    #生成正态分布采样
    noise = torch.randn_like(x_0)
    #得到均值方差
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    #根据x0求xt
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声



# 拟合逆扩散过程高斯分布模型——拟合逆扩散时的噪声


#自定义神经网络
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(in_feature, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, in_feature),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        #         x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x

#编写训练误差函数
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机的时刻t，t变得随机分散一些，一个batch size里面覆盖更多的t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)# t的形状（bz）
    t = t.unsqueeze(-1)# t的形状（bz,1）

    # x0的系数，根号下(alpha_bar_t)
    a = alphas_bar_sqrt[t]

    # eps的系数,根号下(1-alpha_bar_t)
    aml = one_minus_alphas_bar_sqrt[t]

    # 生成随机噪音eps
    e = torch.randn_like(x_0)

    # 构造模型的输入
    x = x_0 * a + e * aml

    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))

    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()

#编写逆扩散采样函数

#从xt恢复x0
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)
    #得到均值
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    #得到sample的分布
    sample = mean + sigma_t * z

    return (sample)



class EMA():
    """构建一个参数平滑器"""

    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


if __name__ == "__main__":

    # # 生成一万个点，得到s curve
    # s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
    # s_curve = s_curve[:, [0, 2]] / 10.0
    #
    # print("shape of s:", np.shape(s_curve))
    #
    # data = s_curve.T


    # 演示加噪过程，加噪100步情况
    num_shows = 20

    # 当成一个数据集
    # dataset = torch.Tensor(s_curve).float()
    # 共有10000个点，每个点包含两个坐标
    # 加载 CSV 数据集
    csv_file = './data/single-bulk/GSE110894/GSE110894.csv'  # 替换为你的 CSV 文件路径
    drop_column = 'gene_id'  # 删除列
    custom_dataset = CustomDataset(csv_file, drop_column=drop_column, transform=ToTensor())

    # 创建数据加载器
    batch_size = 64
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)




        # 在这里执行你的训练或评估步骤
        # features 和 labels 是从数据加载器中获取的批量数据

    # 确定超参数的值
    # 开始训练模型，打印loss以及中间的重构效果
    seed = 1234

    print('Training model...')
    batch_size = 128
    # dataset放到dataloader中
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 迭代周期
    num_epoch = 4000
    plt.rc('text', color='blue')
    # 实例化模型，传入一个数
    model = MLPDiffusion(num_steps)  # 输出维度是2，输入是x和step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # epoch遍历
    for t in range(num_epoch):
        # dataloader遍历
        # 使用数据加载器迭代数据
        for idx, batch in enumerate(data_loader):
            features = batch['feature']

            # 定义要取出的部分的大小
            subset_size = 2000  # 例如，取出2000个元素
            # 生成随机索引，用于取出部分数据
            # 生成每个一维数组的随机索引
            random_indices = torch.randint(0, features.shape[1], (features.shape[0], subset_size))
            # 使用随机索引抽取元素并形成新的 Tensor
            subset_data = torch.gather(features, 1, random_indices)
            # 得到loss
            loss = diffusion_loss_fn(model, subset_data, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            # 梯度clip，保持稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        # 每100步打印效果
        if (t % 100 == 0):
            # 根据参数采样一百个步骤的x，每隔十步画出来，迭代了4000个周期，逐渐更接近于原始
            x_seq = p_sample_loop(model, subset_data.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
            print((x_seq[99] - subset_data).square().mean())
