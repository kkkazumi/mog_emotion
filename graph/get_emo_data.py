import numpy as np
import cv2

print("input the name of the figure")
name = input()

img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
data = np.array(img)
data_ud = np.flipud(data)
y_max = data.shape[0]
x_max = data.shape[1]

y,x = np.where(data_ud<255)

res20 = np.polyfit(x,y,20)
#res5 = np.polyfit(x,y,5)
#res4 = np.polyfit(x,y,4)
#res3 = np.polyfit(x,y,3)
y20 = np.poly1d(res20)(x) #<-fitted function
#y5 = np.poly1d(res5)(x) #<-fitted function
#y4 = np.poly1d(res4)(x) #<-fitted function
#y3 = np.poly1d(res3)(x) #<-fitted function

import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(x,y)
plt.scatter(x,y20)
#plt.scatter(x,y5)
#plt.scatter(x,y4)
#plt.scatter(x,y3)
plt.xlim(0,x_max)
plt.ylim(0,y_max)
plt.show()
