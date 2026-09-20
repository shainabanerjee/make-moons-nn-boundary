# Neural Network Decision Boundary on Moons Dataset

A Python project that uses a neural network built with Keras/TensorFlow to learn and visualize a nonlinear decision boundary on a two-moons dataset, obtained from scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).

The project generates synthetic data, trains a small neural network to classify the two classes, and then visualizes the decision boundary learned by the model.

## Demo

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/6781225e-dbf7-4db6-9e88-39ca12cda632" />

The program produces two plots:

1. **Training Data** — the generated two-moons dataset
2. **AI Decision Boundary** — the same data with an added decision boundary using neural network's predictions

The decision boundary is the contour where the model's predicted probability is approximately `0.5`.

## How It Works

The dataset is generated using `make_moons` from scikit-learn:

```python
X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
```

This creates 500 samples with some random noise (0.3 - can be increased or decreased for experimenting) to make the classification problem less trivial.

The neural network has the following architecture:

```text
Input (2 features) -> Dense(4, tanh) -> Dense(2, tanh) -> Dense(4, tanh) -> Dense(1, sigmoid) -> Binary classification
```

The model is trained using:

* **Optimizer:** Adam
* **Loss:** Binary Cross-Entropy
* **Metric:** Accuracy
* **Epochs:** 200

After training, the program evaluates the model across a grid of points covering the plot. These predictions are used to draw the learned decision boundary.

## Requirements

* Python
* NumPy
* Matplotlib
* scikit-learn
* TensorFlow
* Keras

## Installation

Clone the repository:

```bash
git clone https://github.com/shainabanerjee/make-moons-nn-boundary.git
cd make-moons-nn-boundary
```

Install the dependencies:

```bash
pip install numpy matplotlib scikit-learn tensorflow keras
```

## Running the Project

Run the Python file:

```bash
python draw_boundary.py
```

The program will first display the generated training data.

After closing that plot, the neural network will train for 200 epochs (can be modified for experimentation) and then display the learned decision boundary.

### Avoiding the First Plot

The code contains:

```python
plt.show()
```

If you want the program to move directly to training without displaying the original dataset first, comment out the following line using a preceding '#':

```python
# plt.show()
```

The final plot will still display the learned decision boundary, as shown above, showcasing decent success at segregating the red dots from blue dots and creating a non-linear boundary.

## Working Mechanism

- The two-moons dataset is not linearly separable, meaning a straight line cannot cleanly separate the two classes. Therefore, the neural network creates a non-linear boundary and it is able to do that because the network contains multiple layers with tanh activation functions.

- After 3 hidden layers , a sigmoid layer is used which acts as the output layer. The final layer uses ``layers.Dense(1, activation="sigmoid")``. The sigmoid activation produces a value between 0 and 1, which can be interpreted as the model's estimated probability that a point belongs to one of the classes.

The decision boundary is drawn at: levels=[0.5]

So points with predictions below approximately 0.5 fall on one side of the boundary, while points above 0.5 fall on the other.

- The final visualization contains:
  - Red and blue dots plotted as seen in the original dataset
  - Colored regions representing the model's predictions
  - A black line representing the 0.5 decision boundary


