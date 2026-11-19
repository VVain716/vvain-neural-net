import numpy as np
import mlx.core as mx
import parser
import tqdm

class Conv2D:
  def __init__(self, num_filters, in_channels, filter_size, learning_rate=0.01):
    self.num_filters = num_filters
    self.in_channels = in_channels
    self.filter_size = filter_size
    self.biases = mx.zeros(num_filters, dtype=mx.float32)
    self.learning_rate = learning_rate
    
    self.filters = mx.random.normal((self.num_filters, self.in_channels, self.filter_size, self.filter_size)) * mx.sqrt(2.0 / (self.in_channels * self.filter_size * self.filter_size))
    self.filters = self.filters.astype(mx.float32)

    self.prev_input_shape = None
    self.prev_output = None
    self.X_col = None
  def forward(self, img):
    self.prev_input_shape = img.shape
    B, C, H, W = img.shape
    out_h = H - self.filter_size + 1
    out_w = W - self.filter_size + 1
    window_view = mx.as_strided(img, (B, C, out_h, out_w, self.filter_size, self.filter_size), (C*H*W, H*W, W, 1, W, 1))
    X_col = window_view.transpose(1, 4, 5, 0, 2, 3).reshape(self.in_channels * self.filter_size * self.filter_size, img.shape[0] * window_view.shape[2] * window_view.shape[3])
    self.X_col = X_col
    W_col = self.filters.reshape(self.num_filters, self.in_channels * self.filter_size * self.filter_size)
    val = W_col @ X_col + self.biases.reshape(self.num_filters, 1)
    val = val.reshape(self.num_filters, img.shape[0], window_view.shape[2], window_view.shape[3]).transpose(1, 0, 2, 3)
    output = mx.maximum(0.0, val) # RELU
    self.prev_output = output
    return output
  def backward(self, x):
    B = x.shape[0]
    x = x * (self.prev_output > 0)
    x = x.transpose(1, 0, 2, 3)
    _, _, out_height, out_width = x.shape
    d_out = x.reshape(self.num_filters, B * out_height * out_width)
    grad_biases = mx.sum(d_out, axis=1)
    grad_filts = d_out @ self.X_col.T
    grad_filts = grad_filts.reshape(self.filters.shape)
    W_col = self.filters.reshape(self.num_filters, self.in_channels * self.filter_size * self.filter_size)
    dx_col = W_col.T @ d_out
    dx_col = dx_col.reshape(self.in_channels, self.filter_size, self.filter_size, B, out_height, out_width).transpose(3, 0, 1, 2, 4, 5)
    output = mx.zeros(self.prev_input_shape)
    for i in range(self.filter_size):
      for j in range(self.filter_size):
        output = output.at[:, :, i:i+out_height, j:j+out_width].add(dx_col[:, :, i, j, :, :])

    self.filters -= self.learning_rate * grad_filts
    self.biases -= self.learning_rate * grad_biases

    return output

class MaxPool2D:
  def __init__(self, pool_size=2, stride=2):
    self.pool_size = pool_size
    self.stride = stride

    self.last_input = None
    self.last_output = None

    self.last_window_view = None
  def forward(self, img):
    self.last_input = img
    B, channels, input_height, input_width = img.shape
    output_height = (input_height - self.pool_size) // self.stride + 1
    output_width = (input_width - self.pool_size) // self.stride + 1
    window_view = mx.as_strided(img, (B, channels, output_height, output_width, self.pool_size, self.pool_size), (channels * input_height * input_width, input_height * input_width, self.stride * input_width, self.stride, input_width, 1))
    self.last_window_view = window_view
    self.last_output = mx.max(self.last_window_view, axis=(4, 5)) # RELU

    return self.last_output
  def backward(self, x):
    _, _, H_out, W_out = x.shape
    output = mx.zeros(self.last_input.shape, dtype=mx.float32)
    grad_windows = (self.last_window_view == self.last_output[:, :, :, :, None, None]) * x[:, :, :, :, None, None]
    for i in range(self.pool_size):
      for j in range(self.pool_size):
        output = output.at[:, :, i:i+H_out*self.stride:self.stride, j:j+W_out*self.stride:self.stride].add(grad_windows[:, :, :, :, i, j])

    return output

class Flatten:
  def __init__(self):
    self.last_shape = None

  def forward(self, img):
    self.last_shape = img.shape
    return img.reshape(img.shape[0], -1)
  def backward(self, x):
    return x.reshape(self.last_shape)

class Dense:
  def __init__(self, in_features, out_features, activation, learning_rate=0.01):
    self.in_features = in_features
    self.out_features = out_features
    self.weights = mx.random.normal((self.out_features, self.in_features)) * mx.sqrt(2.0 / self.in_features)
    self.biases = mx.zeros(out_features)
    self.last_input = None
    self.last_output = None
    self.activation = activation
    self.learning_rate = learning_rate
  def forward(self, x):
    self.last_input = x
    output = x @ self.weights.T + self.biases
    if self.activation == "relu":
      output = mx.maximum(0.0, output) #relu
    else:
      output = mx.exp(output - mx.max(output, axis=1, keepdims=True)) / mx.sum(mx.exp(output - mx.max(output, axis=1, keepdims=True)), axis=1, keepdims=True)
    self.last_output = output
    return output
  def backward(self, x):
    d_z = None
    if self.activation == 'relu':
      d_z = x * (self.last_output > 0)
    else:
      d_z = x

    weights_grad = d_z.T @ self.last_input
    bias_grad = mx.sum(d_z, axis=0)
    backward_pass = d_z @ self.weights
    self.weights -= self.learning_rate * weights_grad # Gradient Descent
    self.biases -= self.learning_rate * bias_grad # Grad Descent
    return backward_pass

class CrossEntropyLoss:
  def __init__(self, drift, out_nodes):
    self.drift = drift
    self.out_nodes = out_nodes

  def forward(self, x, correct):
    row_idx = mx.arange(x.shape[0])
    return -mx.mean(mx.log(x[row_idx, correct] + self.drift)) # CE LOSS
  
  def backward(self, x, correct):
    y = mx.zeros((x.shape[0], self.out_nodes))
    y = y.at[mx.arange(x.shape[0]), correct].add(1)
    return (x - y ) / x.shape[0]
