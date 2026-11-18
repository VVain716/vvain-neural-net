import cv2
import numpy as np
import os
import random

def parse_image(path):
  img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    return None
  img = cv2.resize(img, (64, 64))
  arr = np.asarray(img)
  return arr.astype(np.float32) / 255.0

def get_data(dir):
  label_dict = {}
  res = []
  for idx, label in enumerate(os.listdir(dir)):
    label_dict[label] = idx
    print(label)
    for f in os.listdir(os.path.join(dir, label)):
      path = os.path.join(dir, label, f)
      img = parse_image(path)
      if img is None:
        continue
      res.append((img, idx))
  random.shuffle(res)
  return res, label_dict


if __name__ == "__main__":
  training = get_data("data/training")
  print(training[0])


