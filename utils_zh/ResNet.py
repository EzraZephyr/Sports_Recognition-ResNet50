import torch.nn as nn
import torchvision


def ResNet():

    model = torchvision.models.resnet50(pretrained=True)
    # 构建ResNet50模型 并且需要将pretrained置为True
    # 用于加载预训练模型的权重 否则将不会加载权重

    for name, param in model.named_parameters():
        # 遍历模型的所有参数和名称
        # 因为这里要冻结前75%的层 而ResNet50有四个卷积层块
        # 所以我们需要冻结前三个 只是用第四个卷积层块和全连接层
        if name.startswith('layer4') or name.startswith('fc'):
            param.requires_grad = True
            # 如果最后一个卷积层块和全连接层 则设置该层梯度需要更新
        else:
            param.requires_grad = False
            # 禁用前面被冻结的层的梯度更新

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)
    # 获取模型最后一个全连接层 把输出特征数改为我们训练集的类别数量100

    return model