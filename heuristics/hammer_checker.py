#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import datetime
import matplotlib.pyplot as plt

TIME_COL=6

GRAPH_RANGE=150
SKIP_RANGE=10

CHECK_COL=1

label = ['a']*GRAPH_RANGE

class Hammer_check:
  def __init__(self, data,num_lines):
    self.data=data
    self.i = 0
    self.num_lines = num_lines
    self.plt_array=np.zeros((3,GRAPH_RANGE))
    self.plt_array_diff=np.zeros((3,GRAPH_RANGE))
    self.hit_t = 0

    self.fig = plt.figure(figsize=(10, 4))
    self.ax = self.fig.add_subplot(1,1,1)

  def view_graph(self):
    # at this point, it doesnt work
    if(self.i%15==0):
      self.ax.set_ylim(-15,15)
      self.ax.plot(range(GRAPH_RANGE),self.plt_array[1,:],label="imu")
      self.ax.plot(range(GRAPH_RANGE),self.plt_array_diff[1,:],label="imu diff")

      #ax.legend()
      plt.pause(.01)
      plt.clf()

  def get_data(self):
    self.plt_array[:,:-1] = self.plt_array[:,1:]
    self.plt_array_diff[:,:-1] = self.plt_array_diff[:,1:]
    self.plt_array[:,-1]=self.data[self.i,CHECK_COL]
    self.plt_array_diff[:,-1]=self.data[self.i,CHECK_COL]-self.data[self.i-1,CHECK_COL]
    self.i+=1

  def hit_checker(self):
    if abs(self.plt_array_diff[1,-1]) > 1.5:
      if abs(self.i-self.hit_t)>300:
        self.hit_t = self.i #hit_t; last hit time

        hit_time = datetime.datetime(year=2019,month=11,day=1,minute=int(self.data[self.i,TIME_COL]),second=int(self.data[self.i,TIME_COL+1]),microsecond=int(self.data[self.i,TIME_COL+2]))
        print("__________________________________hit: ",hit_time)
      else:
        hit_time = None
    else:
      hit_time = None
    return hit_time


  def main(self):

    for i in range(0,self.num_lines,SKIP_RANGE):
      self.get_data()
      #print i, data[i,6],data[i,7],data[i,8]
      #self.view_graph()
      self.hit_checker()

      #split_t_count+=1

if __name__ == '__main__':
  data=np.loadtxt("imu_test.csv",delimiter=",")
  num_lines = sum(1 for line in open("./imu_test.csv"))
  hmr = Hammer_check(data,num_lines)
  hmr.main()

#removed_split_time_container = np.zeros((3,split_t_count+1))
#removed_split_time_container = split_time_container[:,:split_t_count]
#np.savetxt(dir_path+str(name)+"/split_time_test2.csv",X=removed_split_time_container.T,delimiter=",",fmt="%d")
#np.savetxt(dir_path+str(name)+"/split_time_test2.csv",X=split_time_container.T,delimiter=",",fmt="%d")
