import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from utils_en.data_loader import data_loader
from utils_en.ResNet import ResNet


def pred(image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet().to(device)
    # Load the model architecture

    model.load_state_dict(torch.load("./model/model.pt", map_location=device))
    # Load the model parameters

    model.eval()
    # Set the model to evaluation mode

    with torch.no_grad():
        image = image.to(device)
        out = model(image)
        _, predicted = torch.max(out, 1)
        return predicted.item()
        # Return the index of the predicted label

def load_image(image_path):

    _, _, index_to_class = data_loader()
    image = Image.open(image_path).convert('RGB')
    # Convert the input image to RGB format

    image_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Perform image preprocessing to match the model training

    image = image_transform(image).unsqueeze(0)
    # Preprocess the image and add a batch dimension

    predict = pred(image)

    return index_to_class[predict]
    # Return the final prediction to the GUI
