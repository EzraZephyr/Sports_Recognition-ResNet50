import csv
import torch
import time
from torch import nn
from utils_zh.data_loader import data_loader
from utils_zh.ResNet import ResNet

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet().to(device)
    # 如果有可用的cuda设备 则将设备转移到cuda上进行后续训练

    train_loader, test_loader, _ = data_loader()
    # 加载data_loader中我们处理好的数据加载器

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用交叉熵损失函数计算损失并用Adam优化器优化梯度

    train_log = './model/training.log'
    train_csv = './model/training_metrics.csv'
    file = open(train_log, 'w')
    # 创建训练日志和训练指标的文件

    with open(train_csv, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # 打开csv文件进行写入操作
    # 定义文件的列名为'epoch','loss','accuracy'
    # 创建writer对象并将列名写入标题行

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        start = time.time()
        # 记录这一轮训练的总损失和起始时间

        model.train()
        # 将模型设定为训练模式

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # 设置需要训练的图片和标签的设备

            optimizer.zero_grad()
            # 清空梯度

            outputs = model(images)
            # 向前传播

            loss = criterion(outputs, labels)
            # 计算损失

            loss.backward()
            # 向后传播

            optimizer.step()
            # 更新模型参数

            total_loss += loss.item()
            # 计算总损失

        message = 'Epoch:{}, Loss:{:.4f}, Time:{:0.2f}, '.format(epoch+1, total_loss / len(train_loader), time.time() - start)
        # 记录本次训练的信息

        correct = 0
        total = 0
        # 记录每一批的正确个数和总个数

        model.eval()
        # 将模型设为评估模式

        with torch.no_grad():
            # 禁用梯度计算

            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                # 向前传播

                _, predicted = torch.max(outputs.data, 1)
                # 返回概率最大的类别的索引

                correct += (predicted==labels).sum().item()
                total += labels.size(0)
                # 计算这一批的正确总数和训练样本次数

        message += 'Accuracy: {:.3f}%'.format(100*correct/total)
        file.write(message+'\n')
        print(message)
        # 将正确率加到上面的信息中 并且记录日志和输出

        with open(train_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch+1, 'loss': total_loss / len(train_loader), 'accuracy': 100*correct/total})
        # 打开csv文件进行追加 并将训练轮数 损失值和正确率沿列名写入

    file.close()
    torch.save(model.state_dict(), './model/model.pt')
    # 关闭日志文件并保存模型


