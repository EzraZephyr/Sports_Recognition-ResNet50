from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import \
    Compose, Resize, RandomRotation, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize


def data_loader():

    train_file = './data/train/'
    test_file = './data/test/'

    train_transform = Compose([
        Resize((224, 224)),
        # 先将图片大小调整为224 224的ResNet固定输入格式

        RandomRotation(0.1),
        # 随机旋转图像 旋转弧度为-0.1 - 0.1之间

        RandomHorizontalFlip(0.5),
        # 随机水平翻转图像 概率为0.5

        RandomResizedCrop(size=(224,224),scale=(0.9,1.1)),
        # 将图像随机变换到原来的0.9倍到1.1倍之间
        # 然后裁剪成224 224

        ToTensor(),
        # 归一化为张量

        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 将经过上述处理的图片再次进行标准化 以加快模型收敛速度
        # 和提高模型在不同数据分布间的泛化能力

    ])

    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 将测试集的图像也经过非改变图片样式的处理

    train_data = ImageFolder(train_file, transform=train_transform)
    test_data = ImageFolder(test_file, transform=test_transform)
    # 通过ImageFolder打开图片 根据root为其实路径 打开子文件并创建字典 从0开始设定目标值
    # 再打开每一张图片的时候进行上面设定的transform操作

    class_to_idx = train_data.class_to_idx
    index_to_class = {idx:cla for cla,idx in class_to_idx.items()}
    # dataset.class_to_idx储存的是以子文件名称为键值的字典
    # 为了方便后续答案显示 需要使目标值为键值 调转一下
    # print(class_to_idx) {'air hockey': 0, 'ampute football': 1, 'archery': 2, ...
    # print(idx_to_class) {0: 'air hockey', 1: 'ampute football', 2: 'archery', ...

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    # 创建数据集加载对象 以每32个为一组进行训练或测试
    # 训练集需要打乱顺序 而测试集不用

    return train_loader, test_loader, index_to_class
