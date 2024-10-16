import numpy as np

def rescale_image(im):
  if np.max(im) <= 1.0:
    im = im*255
  return im

def softmax_stable(x, T=1):
  return (np.exp((x - np.max(x))/T) / np.exp((x - np.max(x))/T).sum())

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def tanh(x, B=2):
  return (np.exp(B*x)-1)/(np.exp(B*x)+1)

def fill_placeholders(list_, objects_dict):
  filled_list = list()
  for item in list_:
    filled_item = item
    for k,v in objects_dict.items():
      filled_item = filled_item.replace(k, v)
    filled_list.append(filled_item)
  return filled_list
