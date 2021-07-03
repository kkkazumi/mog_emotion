import numpy as np
import cv2

DIM_POLY = 20

print("input the name of the figure")
name = input()

img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
data = np.array(img)
data_ud = np.flipud(data)
y_max = data.shape[0]
x_max = data.shape[1]

y,x = np.where(data_ud<255)

err = np.empty(DIM_POLY)

for i in range(DIM_POLY):
  res = np.polyfit(x,y,i)
  yy = np.poly1d(res)(x)
  err[i] = np.sum(abs(y-yy))

print(np.argmin(err))
min_id = np.argmin(err)

res_min = np.polyfit(x,y,min_id)
y_min = np.poly1d(res_min)(x) #<-fitted function

import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(x,y)
plt.scatter(x,y_min)
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.show()
