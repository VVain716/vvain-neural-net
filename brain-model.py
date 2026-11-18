from utility import Conv2D, MaxPool2D, Dense, Flatten, CrossEntropyLoss
from parser import get_data, parse_image
from tqdm.auto import tqdm
import numpy as np
class MNISTModel():
  def __init__(self, layers, loss, img_shape):
    self.layers = layers
    self.loss = loss
    self.img_shape = img_shape
  def forward(self, img):
    x = img
    for i in range(len(self.layers)):
      x = self.layers[i].forward(x)
    return x
  def backward(self, x):
    # x is the loss
    for i in range(len(self.layers)):
      x = self.layers[len(self.layers) - i - 1].backward(x)
    return x

  def train_step(self, img, label):
    x = self.forward(img)
    l = self.loss.forward(x, label)
    grad = self.loss.backward(x, label)
    self.backward(grad)
    return l

  def train_loop(self, data, epochs, label_dict):
    for epoch in range(epochs):
      loss = 0
      imgs = 0
      for img, label in tqdm(data):
        img = np.reshape(img, self.img_shape)
        l = self.train_step(img, label)
        loss += l
        imgs += 1

      print(f"EPOCH [{epoch + 1}/{epochs}] LOSS: {loss / imgs}")
  def save(self, path):
    params = {}
    for i, layer in enumerate(self.layers):
      if hasattr(layer, "filters"):
        params[f"layer_{i}_filters"] = layer.filters
        params[f"layer_{i}_biases"] = layer.biases
      elif hasattr(layer, "weights"):
        params[f"layer_{i}_weights"] = layer.weights
        params[f"layer_{i}_biases"] = layer.biases
    np.savez(path, **params)

  def load(self, path):
    with np.load(path) as data:
      for i, layer in enumerate(self.layers):
        if hasattr(layer, "filters"):
          layer.filters = data[f"layer_{i}_filters"]
          layer.biases = data[f"layer_{i}_biases"]
        elif hasattr(layer, "weights"):
          layer.weights = data[f"layer_{i}_weights"]
          layer.biases = data[f"layer_{i}_biases"]

  def predict(self, img):
    img = np.reshape(img, self.img_shape)
    return self.forward(img)

  def test(self, test_set, label_dict):
    correct = 0
    total = 0
    for img, label in tqdm(test_set):
      prediction = self.predict(img)
      if np.argmax(prediction) == label:
        correct += 1
      total += 1
    return correct / total
if __name__ == "__main__":
  # Define Model
  model = MNISTModel([
      Conv2D(8, 1, 3),
      MaxPool2D(2, 2),
      Conv2D(16, 8, 3),
      MaxPool2D(2, 2),
      Conv2D(32, 16, 3),
      MaxPool2D(2, 2),
      Flatten(),
      Dense(1152, 64, "relu"),
      Dense(64, 4, "softmax")
    ],
    CrossEntropyLoss(10 ** -6, 4),
    (1, 64, 64)
    )
  training, label_dict = get_data("brain-data/Training")
  # model.train_loop(training, epochs=3, label_dict=label_dict)
  model.load("brain-model1test-8epochs.npz")
  model.train_loop(training, epochs=4, label_dict=label_dict)
  model.save("brain-model1test-12epochs.npz")
  testing, label_dict = get_data("brain-data/Testing")
  results = model.test(testing, label_dict=label_dict)
  print(f"ACC: {results:.2%}")