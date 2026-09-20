import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from tensorflow import keras
from keras import layers

X, y = make_moons(n_samples=500, noise=0.3, random_state=0)

#drawing dots using matplotlib
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.title("Training Data")

# Comment out the line below before running if you wish to avoid displaying the original plot before the decision boundary
plt.show()


#small neural network 
model = keras.Sequential([
layers.Dense(4, activation="tanh"), 
layers.Dense(2, activation="tanh"),
layers.Dense(4, activation="tanh"), 
layers.Dense(1, activation="sigmoid") 
])


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=200, verbose=1)

# numpy grid
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5,2, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

predictions = model.predict(grid).reshape(xx.shape)

# drawing decision boundary using matplotlib
plt.contourf(xx, yy, predictions, alpha=0.3, cmap="coolwarm")
plt.contour(xx, yy, predictions, levels=[0.5], colors="black")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("AI Decision Boundary")
plt.show()
