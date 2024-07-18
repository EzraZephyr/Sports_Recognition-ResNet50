import torch
from PIL import Image
from torchvision.transforms import \
    Compose,ToTensor,Resize, Normalize
from utils_zh.data_loader import data_loader
from utils_zh.ResNet import ResNet


def pred(image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    # 加载模型的框架

    model.load_state_dict(torch.load("./model/model.pt",map_location=device))
    # 加载模型的参数

    model.eval()
    # 加载模型并设为评估模式

    with torch.no_grad():

        image = image.to(device)
        out = model(image)
        _, predicted = torch.max(out, 1)
        return predicted.item()
        # 返回答案值索引

def load_image(image_path):

    _,_,index_to_class = data_loader()
    image = Image.open(image_path).convert('RGB')
    # 将传入的图片调整为RGB格式

    image_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 进行图片预处理操作 以符合训练模型

    image = image_transform(image).unsqueeze(0)
    # 将图片预处理 并且增加一维批次值

    predict = pred(image)

    return index_to_class[predict]
    # 将最后的答案返回给GUI


