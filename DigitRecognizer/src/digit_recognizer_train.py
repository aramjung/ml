import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from digit_torch import DigitDataset, SimpleCNN
from sklearn.model_selection import train_test_split
from torchvision import transforms

train_df = pd.read_csv('../data/train.csv')

# valules convert dataframe to numpy.  Numpy is expected by pytorch.  
# convert the data type to np.float32 - more memory efficient than default int64/float64.  tensors by default expect float32.  GPU's are optimized for float32
X_train = train_df.drop('label', axis=1).values.astype(np.float32)
# pytorch expects int64 - compatibility
Y_train = train_df['label'].values.astype(np.int64)

# normalize pixel values to 0-1 for faster training (convergence) and stable gradient
# X_train /= 255.0

X_train_split, X_val_split, Y_train_split, Y_val_split = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# data augmentation
train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),  # normalize 0 to 1
    transforms.Normalize((0.5,), (0.5,))  # normalize to -1 to 1 
])

train_dataset = DigitDataset(X_train_split, Y_train_split, train_transforms)
val_dataset = DigitDataset(X_val_split, Y_val_split)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

model = SimpleCNN()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # cross entropy loss log function / research again
print("About to initialize optimizer with:", model)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# This will halve the learning rate every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# training and validation loop
epochs = 80
patience = 7  # early stopping patience
no_improve_epochs = 0

best_accuracy = 0
best_loss = float('inf')
best_accuracy_model_state = None 
best_val_loss_model_state = None 

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Training accuracy
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)
    
    train_accuracy = correct_train / total_train
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs},  Train Accuracy: {train_accuracy:.4f}, Training Loss: {avg_train_loss:.4f}")

    # validation accuracy
    # 1. save the model with the best validation loss instead
    # 2. add regularization
    # 3. abort epochs if model does not improve
    # 4. study notebooks from others 
    model.eval()
    correct = 0 
    total = 0
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            # validation loss 
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)

    if val_accuracy >= best_accuracy:
        best_accuracy = val_accuracy
        best_accuracy_model_state = model.state_dict()
    # Early stopping logic
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss 
        best_val_loss_model_state = model.state_dict()
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    
    print(f'Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')

    if no_improve_epochs >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

    # 🔽 Step the scheduler after each epoch
    scheduler.step()
# save the best accuracy model state 
torch.save(best_accuracy_model_state, '../models/best_accuracy_model.pth')
# save the best validation loss model state
torch.save(best_val_loss_model_state, '../models/best_val_loss_model.pth')
print('training is complete')

