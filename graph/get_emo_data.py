import numpy as np
import cv2

print("input the name of the figure")
name = input()

img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
data = np.array(img)
data_ud = np.flipud(data)
y,x = np.where(data_ud<255)

res3 = np.polyfit(x,y,3)
y3 = np.poly1d(res3)(x) #<-fitted function


