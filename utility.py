import numpy as np
import parser
import tqdm

class Conv2D:
  def __init__(self, num_filters, in_channels, filter_size, learning_rate=0.01):
    self.num_filters = num_filters
    self.in_channels = in_channels
    self.filter_size = filter_size
    self.biases = np.zeros(num_filters, dtype=np.float32)
    self.learning_rate = learning_rate
    
    self.filters = np.random.randn(self.num_filters, self.in_channels, self.filter_size, self.filter_size) * np.sqrt(2.0 / (self.in_channels * self.filter_size * self.filter_size))  
    self.filters = self.filters.astype(np.float32)

    self.prev_input_shape = None
    self.prev_output = None
    self.X_col = None
  def forward(self, img):
    self.prev_input_shape = img.shape
    window_view = np.lib.stride_tricks.sliding_window_view(img, (self.filter_size, self.filter_size), axis=(1, 2))
    X_col = window_view.transpose(0, 3, 4, 1, 2).reshape(self.in_channels * self.filter_size * self.filter_size, window_view.shape[1] * window_view.shape[2])
    self.X_col = X_col
    W_col = self.filters.reshape(self.num_filters, self.in_channels * self.filter_size * self.filter_size)
    val = W_col @ X_col + self.biases.reshape(self.num_filters, 1)
    output = val.reshape(self.num_filters, window_view.shape[1], window_view.shape[2])
    output = np.maximum(0.0, output) # RELU
    self.prev_output = output
    return output
  def backward(self, x):
    x *= (self.prev_output > 0)
    _, out_height, out_width = x.shape
    d_out = x.reshape(self.num_filters, out_height * out_width)
    grad_biases = np.sum(d_out, axis=1)
    grad_filts = d_out @ self.X_col.T
    grad_filts = grad_filts.reshape(self.filters.shape)
    W_col = self.filters.reshape(self.num_filters, self.in_channels * self.filter_size * self.filter_size)
    dx_col = W_col.T @ d_out
    dx_col = dx_col.reshape(self.in_channels, self.filter_size, self.filter_size, out_height, out_width)
    output = np.zeros(self.prev_input_shape)
    for i in range(self.filter_size):
      for j in range(self.filter_size):
        output[:, i:i+out_height, j:j+out_width] += dx_col[:, i, j, :, :]

    self.filters -= self.learning_rate * grad_filts
    self.biases -= self.learning_rate * grad_biases

    return output
        


class MaxPool2D:
  def __init__(self, pool_size=2, stride=2):
    self.pool_size = pool_size
    self.stride = stride

    self.last_input = None
    self.last_output = None
  def forward(self, img):
    self.last_input = img
    channels, input_height, input_width = img.shape
    output_height = (input_height - self.pool_size) // self.stride + 1
    output_width = (input_width - self.pool_size) // self.stride + 1
    output = np.zeros((channels, output_height, output_width), dtype=np.float32)
    for ch in range(channels):
      for row in range(0, input_height - self.pool_size + 1, self.stride):
        for col in range(0, input_width - self.pool_size + 1, self.stride):
          val = np.max(img[ch, row:row+self.pool_size, col:col+self.pool_size])
          output[ch, row // self.stride, col // self.stride] = val
    self.last_output = output
    return output
  def backward(self, x):
    output = np.zeros(self.last_input.shape, dtype=np.float32)
    channels, input_height, input_width = self.last_input.shape
    for ch in range(channels):
      for row in range(0, input_height - self.pool_size + 1, self.stride):
        for col in range(0, input_width - self.pool_size + 1, self.stride):

          arr = self.last_input[ch, row:row+self.pool_size, col:col+self.pool_size]
          max_row, max_col = np.unravel_index(np.argmax(arr), arr.shape)
          output[ch, row + max_row, col + max_col] += x[ch, row // self.stride, col // self.stride]
    return output

class Flatten:
  def __init__(self):
    self.last_shape = None

  def forward(self, img):
    self.last_shape = img.shape
    return img.flatten()
  def backward(self, x):
    return x.reshape(self.last_shape)

class Dense:
  def __init__(self, in_features, out_features, activation, learning_rate=0.01):
    self.in_features = in_features
    self.out_features = out_features
    self.weights = np.random.randn(self.out_features, self.in_features) * np.sqrt(2.0 / self.in_features)
    self.biases = np.zeros(out_features)
    self.last_input = None
    self.last_output = None
    self.activation = activation
    self.learning_rate = learning_rate
  def forward(self, x):
    self.last_input = x
    output = self.weights @ x + self.biases
    if self.activation == "relu":
      output = np.maximum(0.0, output) #relu
    else:
      output = np.exp(output - np.max(output)) / np.sum(np.exp(output - np.max(output)))
    self.last_output = output
    return output
  def backward(self, x):
    d_z = None
    if self.activation == 'relu':
      d_z = x * (self.last_output > 0)
    else:
      d_z = x

    weights_grad = np.outer(d_z, self.last_input)
    bias_grad = d_z
    backward_pass = self.weights.T @ d_z
    self.weights -= self.learning_rate * weights_grad # Gradient Descent
    self.biases -= self.learning_rate * bias_grad # Grad Descent
    return backward_pass

class CrossEntropyLoss:
  def __init__(self, drift, out_nodes):
    self.drift = drift
    self.out_nodes = out_nodes

  def forward(self, x, correct):
    return -np.log(x[correct] + self.drift) # CE LOSS
  
  def backward(self, x, correct):
    y = np.zeros(self.out_nodes)
    y[correct] = 1
    return x - y
