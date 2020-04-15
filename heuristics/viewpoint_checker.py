#view point checker
import numpy as np
import matplotlib.pyplot as plt

DATA_COL_LEN = 6
TIME_COL_LEN = 3


class Dir_check: 
  def __init__(self, data, window_size):
    self.data = data[:,:DATA_COL_LEN]
    self.time_data = data[:,-TIME_COL_LEN:]
    self.win_size = window_size
    self.i = 0

  def get_data(self):
    self.i = self.i + 1
    return np.mean(self.data[self.i:self.i+self.win_size,:],axis=0)

  def push_history(self):
    data = np.zeros((1,DATA_COL_LEN))
    data[0,:] = self.get_data()
    print(data)
    if(self.i==1):
      self.history = data
    else:
      self.history = np.append(self.history,data,axis=0)

  def view_sight(self):
    self.push_history()
    print(self.i)
    if(self.i>self.win_size):
      x_gaze = self.history[self.i-5:self.i,:3]
      y_gaze = self.history[self.i-5:self.i,3:6]
      plt.ylim(-30,30)
      plt.xlim(-30,30)
      if(self.i%1==0):
        plt.plot(x_gaze[:,0],y_gaze[:,0],marker='o',markersize=2)
        plt.plot(x_gaze[:,1],y_gaze[:,1],marker='o',markersize=2)
        plt.plot(x_gaze[:,2],y_gaze[:,2],marker='o',markersize=2)
        plt.pause(.01)
        plt.clf()

if __name__ == '__main__':
  data = np.loadtxt('gaze_test.csv',delimiter=",")
  i=0
  winsize = 10
  gaze = Dir_check(data,winsize)
  while(i<data.shape[0]):
    gaze.view_sight()
    i=i+1
