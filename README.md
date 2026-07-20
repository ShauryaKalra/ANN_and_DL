# ANN and Deep Learning — Assignments

Assignments for the ANN & Deep Learning course. Each folder contains a standalone implementation.

---

## AB#3 — Feedforward Neural Network (PyTorch)

**Folder:** `ab3_feedforward_nn/`

### What it does
Builds a simple feedforward neural network from scratch using PyTorch to classify Iris flowers into 3 categories (Setosa, Versicolor, Virginica).

### Network Architecture
```
Input (4 features)
    ↓
Hidden Layer 1 — 16 neurons + ReLU
    ↓
Hidden Layer 2 — 8 neurons + ReLU
    ↓
Output — 3 classes (flower types)
```

### Key Concepts Used
| Concept | What it means (simply) |
|---|---|
| **Feedforward** | Data flows in one direction only — input → hidden → output |
| **ReLU** | Activation function that adds non-linearity (replaces negatives with 0) |
| **CrossEntropyLoss** | Measures how wrong the predictions are for multi-class problems |
| **Adam Optimizer** | Adjusts weights during training to reduce loss |
| **Backpropagation** | How the network learns — errors flow backwards to fix weights |

### How to Run

**1. Install dependencies**
```bash
pip install torch scikit-learn
```

**2. Run the script**
```bash
cd ab3_feedforward_nn
python feedforward_nn.py
```

**3. Expected output**
```
Training...

Epoch [ 10/100] | Train Loss: 0.8231 | Test Loss: 0.7654 | Test Accuracy: 73.3%
Epoch [ 20/100] | Train Loss: 0.5412 | Test Loss: 0.5123 | Test Accuracy: 86.7%
...
Epoch [100/100] | Train Loss: 0.1023 | Test Loss: 0.0987 | Test Accuracy: 96.7%

── Final Results ──
Test Accuracy: 96.7%  (29/30 correct)
```

### File Structure
```
ab3_feedforward_nn/
├── feedforward_nn.py   # Main script — network definition + training + evaluation
└── requirements.txt    # Dependencies
```

---

*More assignments will be added here as the course progresses.*
