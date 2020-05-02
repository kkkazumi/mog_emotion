import numpy as np
import cv2

def unit_make(data, data_len):
  tsize=data.shape[0]
  ysize=data.shape[1]

  comp_data = cv2.resize(data, (ysize,data_len))
  return comp_data

if __name__ == "__main__":
  from matplotlib import pyplot as plt
  CHECK_ROW = 2
  data = np.loadtxt('test_mogura.csv',delimiter=",")
  half_data = unit_make(data,500)
  plt.plot(data[:,CHECK_ROW])
  plt.show()
  plt.plot(half_data[:,CHECK_ROW])
  plt.show()
