import csv
import torch
import time
from torch import nn
from utils_en.data_loader import data_loader
from utils_en.ResNet import ResNet

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet().to(device)
    # If there is a CUDA device available, move the device to CUDA for subsequent training

    train_loader, test_loader, _ = data_loader()
    # Load the data loaders processed in data_loader

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Use the cross-entropy loss function to calculate the loss and optimize the gradients with the Adam optimizer

    train_log = './model/training.log'
    train_csv = './model/training_metrics.csv'
    file = open(train_log, 'w')
    # Create files for training logs and training metrics

    with open(train_csv, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # Open the CSV file for writing
    # Define the column names as 'epoch', 'loss', 'accuracy'
    # Create a writer object and write the column names to the header row

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        start = time.time()
        # Record the total loss and start time for this epoch

        model.train()
        # Set the model to training mode

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # Set the device for the images and labels to be trained

            optimizer.zero_grad()
            # Clear the gradients

            outputs = model(images)
            # Forward pass

            loss = criterion(outputs, labels)
            # Calculate the loss

            loss.backward()
            # Backward pass

            optimizer.step()
            # Update the model parameters

            total_loss += loss.item()
            # Calculate the total loss

        message = 'Epoch:{}, Loss:{:.4f}, Time:{:0.2f}, '.format(epoch + 1, total_loss / len(train_loader), time.time() - start)
        # Record the information for this training epoch

        correct = 0
        total = 0
        # Record the number of correct predictions and the total number of samples in each batch

        model.eval()
        # Set the model to evaluation mode

        with torch.no_grad():
            # Disable gradient calculation

            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                # Forward pass

                _, predicted = torch.max(outputs.data, 1)
                # Get the index of the class with the highest probability

                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                # Calculate the total number of correct predictions and
                # the total number of samples in this batch

        message += 'Accuracy: {:.3f}%'.format(100 * correct / total)
        file.write(message + '\n')
        print(message)
        # Add the accuracy to the information above, log it and print it

        with open(train_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch + 1, 'loss': total_loss / len(train_loader), 'accuracy': 100 * correct / total})
        # Open the CSV file in append mode and write the epoch number, loss value,
        # and accuracy under the column names

    file.close()
    torch.save(model.state_dict(), './model/model.pt')
    # Close the log file and save the model
