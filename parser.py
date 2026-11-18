import cv2
import numpy as np
import os
import random

def parse_image(path):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    return None
  arr = np.asarray(img)
  return arr.astype(np.float32) / 255.0

def get_data(dir):
  res = []
  for label in os.listdir(dir):
    print(label)
    for f in os.listdir(os.path.join(dir, label)):
      path = os.path.join(dir, label, f)
      img = parse_image(path)
      if img is None:
        continue
      res.append((img, int(label)))
  random.shuffle(res)
  return res


if __name__ == "__main__":
  training = get_data("data/training")
  print(training[0])


