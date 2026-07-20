"""
AB#3 - Feedforward Neural Network using PyTorch
Dataset: Iris (classify flowers into 3 categories)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. LOAD AND PREPARE DATA ──────────────────────────────────────────────────
iris = load_iris()
X = iris.data        # Features: sepal length, sepal width, petal length, petal width
y = iris.target      # Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# Normalize features (zero mean, unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test  = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test  = torch.LongTensor(y_test)


# ── 2. DEFINE THE NETWORK ─────────────────────────────────────────────────────
class FeedforwardNN(nn.Module):
    """
    A simple 3-layer feedforward neural network:
      Input (4 features) → Hidden Layer 1 (16 neurons) → Hidden Layer 2 (8 neurons) → Output (3 classes)
    """
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),    # Input layer  → Hidden layer 1
            nn.ReLU(),           # Activation function (introduces non-linearity)
            nn.Linear(16, 8),   # Hidden layer 1 → Hidden layer 2
            nn.ReLU(),
            nn.Linear(8, 3)     # Hidden layer 2 → Output (3 flower classes)
        )

    def forward(self, x):
        return self.network(x)


# ── 3. SETUP ──────────────────────────────────────────────────────────────────
model     = FeedforwardNN()
criterion = nn.CrossEntropyLoss()          # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer


# ── 4. TRAIN THE MODEL ────────────────────────────────────────────────────────
print("Training...\n")
epochs = 100

for epoch in range(epochs):
    model.train()

    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass (update weights)
    optimizer.zero_grad()   # Clear old gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update weights

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test)
            test_loss  = criterion(test_preds, y_test)
            correct    = (test_preds.argmax(dim=1) == y_test).sum().item()
            accuracy   = correct / len(y_test) * 100

        print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Test Accuracy: {accuracy:.1f}%")


# ── 5. FINAL EVALUATION ───────────────────────────────────────────────────────
print("\n── Final Results ──")
model.eval()
with torch.no_grad():
    final_preds = model(X_test)
    correct     = (final_preds.argmax(dim=1) == y_test).sum().item()
    accuracy    = correct / len(y_test) * 100
    print(f"Test Accuracy: {accuracy:.1f}%  ({correct}/{len(y_test)} correct)")

    print("\nSample Predictions vs Actual:")
    class_names = iris.target_names
    for i in range(5):
        predicted = class_names[final_preds[i].argmax().item()]
        actual    = class_names[y_test[i].item()]
        print(f"  Predicted: {predicted:12s} | Actual: {actual}")
