import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import os
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np

# Dog and Cat Image Classification with PyTorch
# This script prepares the dataset, defines a model, and trains it for binary classification of Dog and Cat images.
# Dog is labelled as 1 and Cat as 0.

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # Convert image to RGB format
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32) # Convert label to float32 for BCEWithLogitsLoss / dimension (batch_size, 1)
            return image, label
        else:
            return image, os.path.basename(img_path)

# Prepare the dataset
# load training, validation, and test datasets from specified directories
# This function prepares the dataset for training, validation, and testing.
## It loads images from a specified directory, applies transformations, and returns a DataLoader.
# train_enum: 0=train, 1=validation
def load_dataset(data_dir, train=True, label=True, batch_size=32, num_workers=4):
    # Define transformations
    # If training, apply transformations with randomization; otherwise, apply only resizing and normalization
    transform_list = [transforms.Resize((224, 224))]
    if train:
        transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10)
        ]
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)

    # Load image paths and labels
    image_paths = []
    labels = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_file)) # Full path to the image file
                    if label:
                        labels.append(1 if class_name.lower()=='dog' else 0)  # Assuming binary classification (Dog, Cat) // 1 for Dog, 0 for Cat
    if not label:
        labels = None 
    # Create dataset and dataloader
    dataset = CustomDataset(image_paths=image_paths, labels=labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    return dataloader

# Load datasets / all data sets have labels in my case 
train_loader = load_dataset('../data/train', train=True, batch_size=32)
validation_loader = load_dataset('../data/validation', train=False, batch_size=32)
test_loader = load_dataset('../data/test', train=False, batch_size=32)

# define the model
def get_model():
    model = models.resnet18(pretrained=True)  # Load a pre-trained ResNet-18 model
    num_features = model.fc.in_features
    model.fc = nn.Linear(model.fc.in_features, 1)  # Replace the final layer - outputs 1 logit for binary classification
    for param in model.parameters():
        param.requires_grad = True  # Fine-tune all layers
    return model

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model().to(device)  # Assuming binary classification (Dog, Cat)
criterion = nn.BCEWithLogitsLoss() # Loss function: Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate of 0.001
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training function
# This function trains the model using the provided DataLoader, loss function, and optimizer.
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, max_patience=5):
    # Initialize variables for tracking best validation loss and patience for early stopping
    best_validation_loss = float('inf')  # Initialize best validation loss
    patience = 0  # Initialize patience for early stopping
    # Iterate over the number of epochs
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_preds = 0.0
        # Iterate over the training data
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"): # tqdm for progress bar
            images = images.to(device) # Move images to the device (GPU or CPU) 
            labels = labels.to(device).float()  # Convert labels to float for BCEWithLogitsLoss and move labels to the device

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs.view(-1), labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step() # Update weights
            # Update running loss
            running_loss += loss.item() * images.size(0)
            # track accuracy
            preds = torch.sigmoid(outputs).view(-1) > 0.5  # Apply sigmoid and threshold at 0.5 / sigmoid converts raw logits to probabilities
            correct_preds += (preds.int() == labels.int()).sum().item()  # Count correct predictions
        # Calculate accuracy
        labels = labels.cpu()  # Move labels back to CPU for accuracy calculation
        train_accuracy = correct_preds / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/ len(train_loader.dataset):.4f}, Training Accuracy: {train_accuracy:.4f}")
        # Validate the model
        validation_loss = validate_model(model, val_loader, criterion)    

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            patience = 0
            best_validation_loss = validation_loss
            print(f"Validation Loss improved to {best_validation_loss:.4f}, saving model...")
            torch.save(model.state_dict(), 'best_model.pth')  # Save the model if validation loss improves
        else:
            patience += 1
            if patience > max_patience:
                print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_validation_loss:.4f}")
                break   


# Validation function
# This function evaluates the model on the validation set and prints the loss and accuracy. 
# It sets the model to evaluation mode, disables gradient calculation, and computes the loss and accuracy    
def validate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_preds = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images) # forward pass
            loss = criterion(outputs.view(-1), labels) # Calculate loss
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs).view(-1) > 0.5
            correct_preds += (preds == labels).sum().item()
    val_accuracy = correct_preds / len(val_loader.dataset)
    validation_loss = running_loss / len(val_loader.dataset)
    print(f"Validation Loss: {validation_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return validation_loss

# Train the model
train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=10, max_patience=5)

# Test function
# This function evaluates the model on the test set and prints the accuracy and classification report. 
# TODO: compute accuracy and classification report
# It sets the model to evaluation mode, disables gradient calculation, and computes the accuracy and classification report 
def test_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    with torch.no_grad():  # Disable gradient calculation for testing
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()  # Convert labels to float for BCEWithLogitsLoss
            outputs = model(images)  # Forward pass
            preds = torch.sigmoid(outputs).view(-1) > 0.5  # Apply sigmoid and threshold at 0.5

            all_preds.extend(preds.cpu().numpy())  # Move predictions to CPU and convert to numpy array
            all_labels.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy array

    accuracy = accuracy_score(all_labels, all_preds)  # Calculate accuracy
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['Cat', 'Dog']))  # Print classification report

# Test the model
model.load_state_dict(torch.load('best_model.pth'))  # Load the best model
model.to(device)  # Move model to the device (GPU or CPU)
test_model(model, test_loader)

print("Training and testing completed successfully.")
