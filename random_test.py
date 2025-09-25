import torch
import torch.optim as optim
import torch.nn as nn

# 假设我们有一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(128, 10)  # 简单的全连接层

    def forward(self, x):
        return self.fc(x)

# 自定义的学习率调整函数
def custom_adjust_learning_rate(optimizer, epoch, loss_history, acc_history, args):
    # 1. 阶段一：前 75 epoch，学习率不变
    if epoch <= 75:
        lr = args.learning_rate

    # 2. 阶段二：75-150 epoch，逐渐减小学习率（线性衰减）
    elif epoch <= 150:
        progress = (epoch - 75) / (150 - 75)  # 0 ~ 1
        lr = args.learning_rate * (1 - 0.8 * progress)  # 例如衰减到原来的 20%

    # 3. 阶段三：epoch > 150，判断模型是否“停滞”来调整学习率
    else:
        if len(loss_history) >= 5:
            recent_losses = loss_history[-5:]
            recent_accs = acc_history[-5:]

            loss_change = max(recent_losses) - min(recent_losses)
            acc_change = max(recent_accs) - min(recent_accs)

            print(f"loss_change: {loss_change}, acc_change: {acc_change}")

            if loss_change < args.loss_threshold and acc_change < args.acc_threshold:
                lr = optimizer.param_groups[0]['lr'] * args.lr_boost_factor
            else:
                lr = optimizer.param_groups[0]['lr']  # 保持不变
        else:
            lr = optimizer.param_groups[0]['lr']  # 初期不够数据，不变

    # 更新优化器
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# 模拟参数
class Args:
    def __init__(self):
        self.learning_rate = 0.01
        self.loss_threshold = 0.7
        self.acc_threshold = 0.7
        self.lr_boost_factor = 1.1
        self.epochs = 250

# 设置模型和优化器
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
args = Args()

# 初始化历史记录
loss_history = []
acc_history = []

# 模拟训练过程
for epoch in range(1, args.epochs + 1):
    # 模拟一个训练批次损失和准确率
    if epoch % 20 == 0:  # 每20个epoch引入突发变化
        loss = torch.rand(1).item() * 10  # 突然增大的损失
        acc = torch.rand(1).item() * 0.1  # 突然较低的准确率
    else:
        loss = torch.rand(1).item()  # 模拟正常损失
        acc = torch.rand(1).item()  # 模拟正常准确率

    # 每个epoch记录损失和准确率
    loss_history.append(loss)
    acc_history.append(acc)

    # 调用学习率调整函数
    lr = custom_adjust_learning_rate(optimizer, epoch, loss_history, acc_history, args)

    # 打印训练信息
    if epoch % 10 == 0:  # 每10个epoch打印一次
        print(f"Epoch [{epoch}/{args.epochs}] | Loss: {loss:.4f} | Accuracy: {acc:.4f} | Learning Rate: {lr:.6f}")
