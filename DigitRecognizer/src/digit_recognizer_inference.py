import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from digit_torch import DigitDataset, SimpleCNN

test_df = pd.read_csv('../data/test.csv')
X_test = test_df.values.astype(np.float32)
# X_test /= 255.0 transforms engine already normalizes

test_dataset = DigitDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64)

# load the model state and infer
model = SimpleCNN()
model.load_state_dict(torch.load('../models/best_val_loss_model.pth'))

# Predict and Save Submission
model.eval()
all_preds = []
with torch.no_grad():
    for images in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)

all_preds = torch.cat(all_preds).numpy()

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(all_preds) + 1),
    "Label": all_preds
})

submission.to_csv("../data/submission_best_val_loss.csv", index=False)

print('inference is complete')