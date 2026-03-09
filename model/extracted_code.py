import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

print('Essential libraries imported successfully.')
----------------------------------------
plt.figure(figsize=(15, 6))

# 1. Class Distribution
plt.subplot(1, 2, 1)
sns.countplot(x='Class', data=df, hue='Class', palette='viridis', legend=False)
plt.title('Distribution of Transactions (0: Normal, 1: Fraud)')
plt.yscale('log') # Use log scale to see the small count of fraud cases

# 2. Correlation Heatmap
plt.subplot(1, 2, 2)
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()

# Print class counts
print('Class Counts:')
print(df['Class'].value_counts())
print('\nPercentage of Fraudulent Transactions: {:.2f}%'.format(df['Class'].value_counts()[1] / len(df) * 100))
----------------------------------------
from sklearn.model_selection import train_test_split

# 1. Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# 2. Initial train-test split (80/20) with stratification
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Filter training set to contain ONLY 'Normal' transactions (Class 0)
# We use the full training partition to isolate normal cases
train_mask = (y_train_full == 0)
X_train = X_train_full[train_mask]

# 4. Final verification and shapes
print(f"Original dataset shape: {df.shape}")
print(f"Initial training partition (Normal + Fraud): {X_train_full.shape}")
print(f"Final Autoencoder training set (Normal only): {X_train.shape}")
print(f"Testing set shape (Features): {X_test.shape}")
print(f"Testing set shape (Labels): {y_test.shape}")

# Verify Class 1 is excluded from training
print(f"\nFraud cases in training set: {(y_train_full[train_mask] == 1).sum()}")
print(f"Fraud cases in testing set: {(y_test == 1).sum()}")
----------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super(Autoencoder, self).__init__()
        # Encoder: 30 -> 14 -> 7
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 7),
            nn.ReLU()
        )
        # Decoder: 7 -> 14 -> 30
        self.decoder = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = Autoencoder(input_dim=30).to(device)

print(f'Model instantiated on device: {device}')
print(model)
----------------------------------------
import torch.optim as optim

# 1. Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Training configuration
num_epochs = 50
train_losses = []

# 3. Training Loop
model.train()

print('Starting training...')
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        # Get inputs; data is a list [features]
        inputs = data[0].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

print('Finished Training')

# Visualization of Training Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Autoencoder Training Loss Convergence')
plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
----------------------------------------
# Define the file path for saving the model
MODEL_PATH = 'fraud_autoencoder.pth'

# Save the model state dictionary
torch.save(model.state_dict(), MODEL_PATH)

print(f'Model state dictionary saved to {MODEL_PATH}')

# Optional: Provide a way to download the file directly in Colab
from google.colab import files
# files.download(MODEL_PATH) # Uncomment to download automatically
----------------------------------------
import torch.optim as optim

# 1. Re-initialize Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Training configuration
num_epochs = 50
train_losses = []

# 3. Training Loop
model.train()

print('Starting training for 50 epochs...')
for epoch in range(num_epochs):
    running_loss = 0.0
    for data in train_loader:
        # Get inputs from the batch
        inputs = data[0].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass (Autoencoder reconstructs the input)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')

print('Finished Training')

# 4. Visualization of Training Loss (Fixed Indentation)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, color='blue', label='Training Loss')
plt.title('Autoencoder Training Loss Convergence')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
----------------------------------------
model.eval()
reconstruction_errors = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        # Calculate MSE per sample: (input - output)^2 averaged over features
        mse = torch.mean((inputs - outputs)**2, dim=1)
        reconstruction_errors.extend(mse.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

reconstruction_errors = np.array(reconstruction_errors)
true_labels = np.array(true_labels)

# 1. ROC-AUC
fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
roc_auc = auc(fpr, tpr)

# 2. Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(true_labels, reconstruction_errors)

# Plotting
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# 3. Determine Threshold (using 95th percentile of normal reconstructions in test set as a heuristic)
# In a real scenario, we might optimize F1 score.
normal_errors = reconstruction_errors[true_labels == 0]
threshold = np.percentile(normal_errors, 95)
print(f'Chosen Threshold (95th percentile of normal errors): {threshold:.4f}')

# 4. Confusion Matrix
y_pred = [1 if e > threshold else 0 for e in reconstruction_errors]
cm = confusion_matrix(true_labels, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix at Selected Threshold')
plt.show()

from sklearn.metrics import classification_report
print('Classification Report:')
print(classification_report(true_labels, y_pred))
----------------------------------------
