from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import \
    Compose, Resize, RandomRotation, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Normalize


def data_loader():

    train_file = './data/train/'
    test_file = './data/test/'

    train_transform = Compose([
        Resize((224, 224)),
        # First, resize the image to 224x224, which is the fixed input size for ResNet

        RandomRotation(0.1),
        # Randomly rotate the image with a degree range of -0.1 to 0.1 radians

        RandomHorizontalFlip(0.5),
        # Randomly flip the image horizontally with a probability of 0.5

        RandomResizedCrop(size=(224,224), scale=(0.9,1.1)),
        # Randomly scale the image between 0.9 to 1.1 times its original size
        # Then crop it to 224x224

        ToTensor(),
        # Normalize the image to a tensor

        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Standardize the image to speed up model convergence
        # and improve the model's generalization across different data distributions

    ])

    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Apply the same transformations to the test set images, except for those that alter the image style

    train_data = ImageFolder(train_file, transform=train_transform)
    test_data = ImageFolder(test_file, transform=test_transform)
    # Load images using ImageFolder with the root path, create a dictionary of
    # subfolders and assign targets starting from 0 Apply the specified transform
    # operations to each image when loading

    class_to_idx = train_data.class_to_idx
    index_to_class = {idx: cla for cla, idx in class_to_idx.items()}
    # dataset.class_to_idx stores a dictionary with subfolder names as keys
    # For easier result display later, reverse the dictionary to make target values the keys
    # print(class_to_idx) {'air hockey': 0, 'ampute football': 1, 'archery': 2, ...
    # print(idx_to_class) {0: 'air hockey', 1: 'ampute football', 2: 'archery', ...

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    # Create data loaders, batch size of 32 for training or testing
    # Shuffle the training set but not the test set

    return train_loader, test_loader, index_to_class
