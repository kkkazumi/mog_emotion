#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import matplotlib.pyplot as plt

TIME_COL=6 #number of skipped cols
TIME_COL_LEN = 3 #number of imported cols

GRAPH_RANGE=150
SKIP_RANGE=10

CHECK_COL=1

class Hammer_check:
  def __init__(self, data,num_lines):
    self.data=data
    self.i = 0
    self.hit_t = 0
    self.num_lines = num_lines
    self.plt_array=np.zeros((3,GRAPH_RANGE))
    self.plt_array_diff=np.zeros((3,GRAPH_RANGE))
    self.hit_history = np.empty((1,TIME_COL_LEN))

    #self.fig = plt.figure(figsize=(10, 4))
    #self.ax = self.fig.add_subplot(1,1,1)

  def view_graph(self):
    # at this point, it doesnt work
    if(self.i%50==0):
      plt.ylim(-15,15)
      plt.plot(range(GRAPH_RANGE),self.plt_array[1,:],label="imu")
      plt.plot(range(GRAPH_RANGE),self.plt_array_diff[1,:],label="imu diff")

      #ax.legend()
      plt.pause(.01)
      plt.clf()

  def get_data(self):
    self.plt_array[:,:-1] = self.plt_array[:,1:]
    self.plt_array_diff[:,:-1] = self.plt_array_diff[:,1:]
    self.plt_array[:,-1]=self.data[self.i,CHECK_COL]
    self.plt_array_diff[:,-1]=self.data[self.i,CHECK_COL]-self.data[self.i-1,CHECK_COL]
    self.i+=1

  def time_of_hit(self,ref_time):
    if(len(self.hit_history)>1):
      last_hit_time = self.hit_history[-1,:]
      ret_time = ref_time - last_hit_time
      print("time check",ret_time)
    else:
      ret_time = None
    return ret_time

  def hit_checker(self):
    if abs(self.plt_array_diff[1,-1]) > 1.5:
      if abs(self.i-self.hit_t)>300:
        hit_time=np.zeros((1,TIME_COL_LEN))
        np.set_printoptions(suppress=True)
        hit_time[:,0]=self.data[self.i,TIME_COL]
        hit_time[:,1]=self.data[self.i,TIME_COL+1]
        hit_time[:,2]=self.data[self.i,TIME_COL+2]
        #self.time_of_hit(hit_time) #for debug
        if(self.hit_t==0):
          self.hit_history = hit_time
        else:
          self.hit_history = np.append(self.hit_history,hit_time,axis=0)
        print("hit time_check:",self.i,hit_time,"hit_t",self.hit_t)
        self.hit_t = self.i #hit_t; last hit time
        #hit_time = datetime.datetime(year=2019,month=11,day=1,minute=int(self.data[self.i,TIME_COL]),second=int(self.data[self.i,TIME_COL+1]),microsecond=int(self.data[self.i,TIME_COL+2]))
      else:
        hit_time = None
    else:
      hit_time = None
    return hit_time

  def main(self):

    #for i in range(0,self.num_lines,SKIP_RANGE):
    i=0
    while(True):
      self.get_data()
      #print i, data[i,6],data[i,7],data[i,8]
      self.view_graph()
      self.hit_checker()
      i+=1
    print(i)

      #split_t_count+=1

if __name__ == '__main__':
  data=np.loadtxt("imu_test.csv",delimiter=",")
  num_lines = sum(1 for line in open("./imu_test.csv"))
  hmr = Hammer_check(data,num_lines)
  hmr.main()
