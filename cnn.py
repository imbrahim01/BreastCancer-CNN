import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the MLP model
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, out_sz=2, layers=[120, 84]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, layers[0])  # input layer
        self.fc2 = nn.Linear(layers[0], layers[1])  # hidden layer 1
        self.fc3 = nn.Linear(layers[1], out_sz)  # hidden layer 2 to output
    
    def forward(self, X):
        X = X.view(X.size(0), -1)  # Flatten input dynamically
        X = torch.relu(self.fc1(X))  # input layer to hidden layer 1
        X = torch.relu(self.fc2(X))  # hidden layer 1 to hidden layer 2
        X = self.fc3(X)  # hidden layer 2 to output
        return torch.log_softmax(X, dim=1)

# Define the ResNet model
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Assuming binary classification

    def forward(self, x):
        return self.model(x)

# Streamlit app
st.title('Image Classification with MLP or ResNet')

# Model selection
model_choice = st.selectbox('Choose model', ('MLP', 'ResNet'))

# Upload data directories
train_data_dir = st.text_input('Enter path to training data directory')
validation_data_dir = st.text_input('Enter path to validation data directory')
test_data_dir = st.text_input('Enter path to test data directory')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets if paths are provided
if train_data_dir and validation_data_dir and test_data_dir:
    train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
    validation_dataset = datasets.ImageFolder(root=validation_data_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Display data preview
    st.subheader('Data Preview')
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    fig, ax = plt.subplots(figsize=(12, 8))
    img_grid = np.transpose(utils.make_grid(images, nrow=5).numpy(), (1, 2, 0))
    ax.imshow(img_grid)
    ax.axis('off')
    st.pyplot(fig)

    if st.button('Start Training'):
        # Initialize model, criterion, and optimizer
        if model_choice == 'MLP':
            input_dim = images[0].numel()  # Calculate the input dimension dynamically
            model = MultilayerPerceptron(input_dim=input_dim)
        else:
            model = ResNetModel()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 10
        train_accuracy_history = []
        train_loss_history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_accuracy = 100 * correct / total
            train_accuracy_history.append(train_accuracy)
            train_loss_history.append(train_loss / len(train_loader))
        
        # Evaluate model on test data
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_targets.extend(targets.numpy())
                all_predictions.extend(predicted.numpy())

        test_accuracy = 100 * correct / total

        st.write(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

        # Plot accuracy
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs+1), train_accuracy_history, label='Train Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Train Accuracy Over Epochs')
        ax.legend()
        st.pyplot(fig)

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs+1), train_loss_history, label='Train Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Train Loss Over Epochs')
        ax.legend()
        st.pyplot(fig)

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)
