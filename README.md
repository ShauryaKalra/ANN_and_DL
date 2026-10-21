# ANN and Deep Learning

Course assignments for ANN & Deep Learning — BSc Data Science & AI, Christ University Delhi NCR.

All implementations are from scratch or near-scratch. Libraries used: NumPy, PyTorch, Keras.

---

## labs

| lab | topic | notebook |
|-----|-------|---------|
| Lab 1 | Perceptron simulation using NumPy — tested on AND, OR, NAND, XOR gates | `labs/Lab_1.ipynb` |
| Lab 2 | Single neuron from scratch with gradient descent | `labs/lab_2.ipynb` |
| Lab 3 | Feedforward neural network in PyTorch | `labs/Lab_3.ipynb` |
| Lab 4 | Multi-layer perceptron in Keras for multiclass classification | `labs/Lab_4.ipynb` |
| Lab 5 | CNN in Keras for binary image classification | `labs/Lab_5.ipynb` |
| Lab 6 | MLP in Keras for regression (California Housing), activation and loss comparison | `labs/Lab_6.ipynb` |
| Lab 7 | MLP in Keras for 7-class classification (Dry Beans), activation and optimiser comparison | `labs/Lab_7.ipynb` |
| Lab 8 | SimpleRNN text classification on IMDB reviews | `labs/LAB_8.ipynb` |

each lab has a matching README (`Lab_N_README.md`) explaining the concept and code line by line.

---

## assignments

| folder | topic |
|--------|-------|
| `ab3_feedforward_nn/` | Feedforward NN in PyTorch — Iris flower classification (96.7% accuracy) |
| `CNN_Animals10_Classification/` | CNN for Animals-10 dataset classification |

---

## how to run

```bash
# notebooks
jupyter notebook labs/Lab_1.ipynb

# python scripts
cd ab3_feedforward_nn
pip install -r requirements.txt
python feedforward_nn.py
```
