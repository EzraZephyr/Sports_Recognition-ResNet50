import torch.nn as nn
import torchvision


def ResNet():

    model = torchvision.models.resnet50(pretrained=True)
    # Build the ResNet50 model and set pretrained to True
    # This loads the pretrained weights, otherwise weights will not be loaded

    for name, param in model.named_parameters():
        # Iterate over all model parameters and names
        # Here we need to freeze the first 75% of the layers, and ResNet50 has four
        # convolutional blocks So we need to freeze the first three,
        # only using the fourth convolutional block and the fully connected layer
        if name.startswith('layer4') or name.startswith('fc'):
            param.requires_grad = True
            # Set gradients to be updated for the last convolutional block and
            # the fully connected layer
        else:
            param.requires_grad = False
            # Disable gradient updates for the frozen layers

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)
    # Get the last fully connected layer of the model and change the output features
    # to the number of classes in our dataset, which is 100

    return model
