import utility
import gpuutility
from parser import get_data, parse_image
from tqdm.auto import tqdm
import random
import numpy as np
import mlx.core as mx
import copy
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

  def train_loop(self, data, epochs, batch_size):
    for epoch in range(epochs):
      random.shuffle(data)
      loss = 0
      imgs = 0
      for i in tqdm(range(0, len(data), batch_size)):
        batched = data[i:i+batch_size]
        img_stack = []
        label_stack = []
        for img, label in batched:
          img_stack.append(img.reshape(self.img_shape))
          label_stack.append(label)
        batched_imgs = np.stack(img_stack)
        labels = np.array(label_stack)
        l = self.train_step(batched_imgs, labels)
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
    img = np.reshape(img, (1, *self.img_shape))
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

class MNISTModelGPU():
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
    to_eval = [l]
    for layer in self.layers:
      if hasattr(layer, "filters"):
        to_eval.extend([layer.filters, layer.biases])
      elif hasattr(layer, "weights"):
        to_eval.extend([layer.weights, layer.biases])
    mx.eval(to_eval)
    return l.item()

  def train_loop(self, data, epochs, batch_size):
    for epoch in range(epochs):
      random.shuffle(data)
      loss = 0
      imgs = 0
      for i in tqdm(range(0, len(data), batch_size)):
        batched = data[i:i+batch_size]
        img_stack = []
        label_stack = []
        for img, label in batched:
          img_stack.append(img.reshape(self.img_shape))
          label_stack.append(label)
        batched_imgs = mx.array(np.stack(img_stack))
        labels = mx.array(label_stack)
        l = self.train_step(batched_imgs, labels)
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
    mx.savez(path, **params)

  def load(self, path):
    data = mx.load(path)
    for i, layer in enumerate(self.layers):
      if hasattr(layer, "filters"):
        layer.filters = data[f"layer_{i}_filters"]
        layer.biases = data[f"layer_{i}_biases"]
      elif hasattr(layer, "weights"):
        layer.weights = data[f"layer_{i}_weights"]
        layer.biases = data[f"layer_{i}_biases"]

  def predict(self, img):
    img = mx.array(img)
    img = mx.reshape(img, (1, *self.img_shape))
    return self.forward(img)

  def test(self, test_set, label_dict):
    correct = 0
    total = 0
    for img, label in tqdm(test_set):
      prediction = self.predict(img)
      if mx.argmax(prediction) == label:
        correct += 1
      total += 1
    return correct / total


def cpu():
  # Define Model
  model = MNISTModel([
      utility.Conv2D(16, 1, 3),
      utility.MaxPool2D(2, 2),
      utility.Conv2D(32, 16, 3),
      utility.MaxPool2D(2, 2),
      utility.Conv2D(64, 32, 3),
      utility.MaxPool2D(2, 2),
      utility.Flatten(),
      utility.Dense(12544, 64, "relu"),
      utility.Dense(64, 4, "softmax")
    ],
    utility.CrossEntropyLoss(10 ** -6, 4),
    (1, 128, 128)
    )
  training, label_dict = get_data("brain-data/Training", 128, 128)
  # model.train_loop(training, epochs=3, label_dict=label_dict)
  model.train_loop(training, epochs=5, batch_size=32)
  model.save("models/brain-models/brain-model3test-5epochs.npz")
  testing, label_dict = get_data("brain-data/Testing", 128, 128)
  results = model.test(testing, label_dict=label_dict)
  print(f"ACC: {results:.2%}")


def gpu():
  model1 = MNISTModelGPU([
        gpuutility.Conv2D(16, 1, 3),
        gpuutility.MaxPool2D(2, 2),
        gpuutility.Conv2D(32, 16, 3),
        gpuutility.MaxPool2D(2, 2),
        gpuutility.Conv2D(64, 32, 3),
        gpuutility.MaxPool2D(2, 2),
        gpuutility.Flatten(),
        gpuutility.Dense(12544, 64, "relu"),
        gpuutility.Dense(64, 2, "softmax")
      ],
      gpuutility.CrossEntropyLoss(10 ** -6, 2),
      (1, 128, 128)
      )
  model2 = copy.deepcopy(model1)
  model3 = copy.deepcopy(model1)
  training, label_dict = get_data("chestxray-data/train", 128, 128)
  testing, label_dict = get_data("chestxray-data/test", 128, 128)

  # model.train_loop(training, epochs=3, label_dict=label_dict)
  model1.train_loop(training, epochs=10, batch_size=16)
  model1.save("models/cxr-models/cxr-model1test-10epochs.npz")
  
  results = model1.test(testing, label_dict=label_dict)

  print(f"ACC: {results:.2%}")
  model2.train_loop(training, epochs=15, batch_size=16)
  model2.save("models/cxr-models/cxr-model1test-15epochs.npz")
  results = model2.test(testing, label_dict=label_dict)
  print(f"ACC: {results:.2%}")

  model3.train_loop(training, epochs=20, batch_size=16)
  model3.save("models/cxr-models/cxr-model1test-20epochs.npz")
  results = model3.test(testing, label_dict=label_dict)
  print(f"ACC: {results:.2%}")

if __name__ == "__main__":
  gpu()