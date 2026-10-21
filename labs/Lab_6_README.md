# Lab 6: Keras MLP for Regression (California Housing)

Notebook: `Lab_6.ipynb`
Course: Artificial Neural Networks and Deep Learning

---

## 1. What are we trying to do?

Predict the median house value of a district in California from 8 numbers about it (income, house age, rooms, population, location and so on). This is **regression**: the output is a number, not a class.

The lab asks one question twice: how much do the **activation function** and the **loss function** change the result?

## 2. The dataset

California Housing, loaded from `sklearn.datasets`. 20,640 districts, 8 features, no missing values. The target `MedHouseVal` is in units of $100k and is capped at 5.0 ($500k), so the highest-value districts all look the same. That cap matters for the loss function comparison.

Split: 70% train, 15% validation, 15% test. Features are standardised (zero mean, unit variance) because they sit on very different scales.

## 3. The model

```
Input (8) -> Dense 128 -> BatchNorm -> Dense 64 -> BatchNorm -> Dense 32 -> Dense 1 (linear)
```

- L2 regularisation (1e-4) on the hidden layers.
- The output layer has **no activation**, because a house price can be any number.
- Adam, learning rate 0.001, batch size 64, up to 150 epochs.
- **Early stopping** on validation loss (patience 15), restoring the best weights. That is why each run stopped between about 30 and 60 epochs.

## 4. The two experiments

**A. Activation (loss fixed to MSE):** ReLU, Sigmoid, Tanh.
**B. Loss (activation fixed to ReLU):** MSE, MAE, Huber.

## 5. Results (test set)

| Run | Activation | Loss | MAE | RMSE | R2 |
|---|---|---|---|---|---|
| A1 | ReLU | MSE | 0.4230 | 0.5899 | 0.7345 |
| A2 | Sigmoid | MSE | **0.3564** | **0.5285** | **0.7868** |
| A3 | Tanh | MSE | 0.3626 | 0.5331 | 0.7831 |
| B2 | ReLU | MAE | 0.3941 | 0.5665 | 0.7552 |
| B3 | ReLU | Huber | 0.4284 | 0.5918 | 0.7327 |

MAE of 0.3564 means the typical error is about $35,600. R2 of 0.79 means the model explains about 79% of the variation in prices.

## 6. What the numbers say

- **Activation:** Sigmoid and Tanh beat ReLU here. Sigmoid was best on every metric. This goes against the usual advice that ReLU wins. A likely reason is that this network is small and shallow (three hidden layers) and uses BatchNorm, so vanishing gradients are not a real problem. House prices also change smoothly with the inputs, which suits a smooth activation. This is one seed per run, so a small gap could partly be luck.
- **Loss:** with ReLU, MAE did better than MSE, and Huber was the weakest. MSE punishes large errors hard, and the $500k cap creates a pile of those, so a loss that ignores them more (MAE) helped.
- **Best model:** Sigmoid + MSE.

## 7. Note on repeat runs

ReLU + MSE was trained twice with the same seed and scored R2 0.7345 (run A1) and 0.7509 (run B1), so one run can move by about 0.016. The activation gap (Sigmoid 0.787 against ReLU 0.735 to 0.751) is bigger than that. The loss gaps are not, so treat the loss ranking as a hint.

The written analysis in the notebook was corrected to match these printed results.

## 8. How to run

Python 3, TensorFlow 2.x, scikit-learn, pandas, matplotlib, seaborn. Open `Lab_6.ipynb` in Jupyter or Colab and run all cells. The dataset downloads through scikit-learn.
