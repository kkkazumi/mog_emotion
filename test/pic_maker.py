import numpy as np
import cv2
import pandas as pd

import os
import matplotlib.pyplot as plt

from get_mood_data import *

filename_label = ['ad_all','imu_all','hap_out','sup_out','ang_out','sad_out','neu_out','gaze_out','head_pose_polate']
limit_error = [1000,1000,50000,50000,50000,50000,50000,50000,50000]
data_col = [8,6,3,3,3,3,3,3,3]

def gauss(x,sig):
  mu = 0
  return np.exp(-x**2/(2*sig**2))

def min_max(x, axis=None):
  min = x.min(axis=axis, keepdims=True)
  max = x.max(axis=axis, keepdims=True)
  result = (x-min)/(max-min)
  return result

def read_files(user_name,data_type):
  data_path="/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/backup/home/kazumi/mogura/"

  filename = data_path + user_name + '/' + filename_label[data_type]+'.csv'
  data = np.loadtxt(filename,delimiter=",")

  return data

def unit_make(data, data_len):
  tsize=data.shape[0]
  ysize=data.shape[1]
  data_len = data.shape[0]
  comp_data = cv2.resize(data, (ysize,data_len))
  return comp_data

def get_time(data_array,target_row):
  data_min = data_array[target_row,-3]
  data_sec = data_array[target_row,-2]
  data_msec = data_array[target_row,-1]
  target_time = data_min,data_sec,data_msec
  return target_time

def check_time(check_array,target_time,order):
  target_min, target_sec, target_msec = target_time
  data=np.where((check_array[:,-3]==target_min)&(check_array[:,-2]==target_sec)&((check_array[:,-1]>target_msec-order)&(check_array[:,-1]<target_msec+order)))
  _mostmin=np.argmin(abs(check_array[data,-1]-target_msec))
  same_time_row=data[0][_mostmin]
  return same_time_row


class Data:
  def __init__(self, username,data_type):
    self.username = username
    self.data_type = data_type #type of data (hap, gaze etc..)
    self.data = read_files(self.username,data_type)

  def check_start(self):
    START_ROW = 1
    start_time_cand= get_time(self.data,START_ROW)
    
    return start_time_cand

  def set_start_data(self, start_time,end_time):
    self.end_time = end_time[0],end_time[1],end_time[2]

    self.end_row = check_time(self.data,self.end_time,limit_error[self.data_type])
    self.start_row = check_time(self.data,start_time,limit_error[self.data_type])

  def get_unit(self,start_time,end_time,shape):
    self.set_start_data(start_time,end_time)
    #print('unit2')
    #print('self.data.shape',self.data.shape)
    #print('check row',self.start_row, self.end_row,data_col,self.data_type)
    unit = unit_make(self.data[self.start_row:self.end_row,:data_col[self.data_type]],shape)
    return unit

#calculating moving average
def average(data,size):
  #import matplotlib.pyplot as plt
  b = np.ones(size)/size
  moving_average = np.zeros_like(data)
  for i in range(data.shape[1]):
    moving_average[:,i] = np.convolve(data[:,i],b,mode='same')
    #plt.plot(data[:,i],label='raw')
    #plt.plot(moving_average[:,i],label='average')
    #plt.legend()
    #plt.show()
  return moving_average

def data2file(data,start_time,end_time,filename,str_part):
    hap,sup,ang,sad,neu=data
    shape = hap.data.shape[0]
    #print('check shape',shape)
    ave_size = 5
    
    half_data = np.hstack((hap.get_unit(start_time,end_time,shape),
    sup.get_unit(start_time,end_time,shape),
    ang.get_unit(start_time,end_time,shape),
    sad.get_unit(start_time,end_time,shape),
    neu.get_unit(start_time,end_time,shape)))

    np.savetxt(filename+'_'+str_part+'.csv',average(half_data,ave_size),delimiter=",")
    #np.savetxt(username+'test_class_'+str(i)+'.csv',half_data,delimiter=",")
    cv2.imwrite(filename+'_'+str_part+'.png',half_data.T)
    return half_data.shape

def add1sec(before_time):
    #print('before time',before_time)
    after_time = np.zeros(3)

    after_time[0] = before_time[0]
    after_time[1] = before_time[1]+1.0
    after_time[2] = before_time[2]
    if after_time[1] >= 60:
      after_time[1] = after_time[1] - 60
      after_time[0] += 1
      if after_time[0]>=60:
        after_time[0] = after_time[0] - 60
    #print('after time',after_time)
    return after_time[0],after_time[1],after_time[2]

def out_all_data(username,start_time=None,end_time = None):
  hap = Data(username,2)
  sup = Data(username,3)
  ang = Data(username,4)
  sad = Data(username,5)
  neu = Data(username,6)

  #TODO: import here to get mogura/hammer sensors and imu data

  if(start_time == None):
    start_time = max(hap.check_start(),
    sup.check_start(),
    ang.check_start(),
    sad.check_start(),
    neu.check_start())

  print("start time",start_time)
  print("end time",end_time)

  qfile_path = '../emo_questionnaire/'+username+'.csv'
  i=0
  data = hap,sup,ang,sad,neu

  output_emo_data = np.zeros(hap.data.shape[0])

  #get emotion data
  if os.path.exists(qfile_path):
    df = pd.read_csv('../emo_questionnaire/'+username+'.csv',header=None)#,delimiter=",",dtype="unicode")
    emo_data = np.zeros((df.shape[0],df.shape[1]-1),dtype=np.int)
    emo_data = df.values[:,:5]
    emo_type = df.values[:,5]
    #emo_data = np.zeros_like(df.values[])
    #emo_data = np.loadtxt('../emo_questionnaire/'+str(username)+'.csv',delimiter=",",dtype="unicode")

    filename = './output/'+username+'_face_test2_class_'+str(i)
    sensor_len = data2file(data,start_time,end_time,filename,'1st')

    if(end_time ==None):
      for i in range(emo_data.shape[0]):
        print('i',i)
        end_time = emo_data[i,2],emo_data[i,3],emo_data[i,4]

        centre_row = check_time(hap.data,end_time,limit_error[hap.data_type])
        print("centre_row",centre_row)

        sigma = check_time(hap.data,add1sec(end_time),limit_error[hap.data_type])-centre_row
        print('sigma',sigma)
        x = np.linspace(-sigma,sigma,sigma*2)
        yline=gauss(x,sigma/3.0)

        output_emo_data[centre_row-sigma:centre_row+sigma] = yline
        plt.plot(output_emo_data)
        plt.show()

    else:
      filename = "../graph/test/test_"+ username+"_joy.png"
      x,y=get_mood(filename)
      _,res=ret_func(x,y)
      st_row = check_time(hap.data,start_time,limit_error[hap.data_type])
      en_row = check_time(hap.data,end_time,limit_error[hap.data_type])

      print(sensor_len)
      mag = sensor_len[0]

      y_array = np.poly1d(res)(x)
      y=np.reshape(y_array,(y_array.shape[0],-1))
      print("yshape",y.shape,max(y),min(y))
      yline = np.array(cv2.resize(y,dsize=(1,mag)))
      resized_mood = yline[:,0]
      print(resized_mood)
      np.savetxt("test_resized_mood.csv",resized_mood,delimiter=",")

    #data2file(data,end_time,add1sec(end_time),filename,'2nd')

    return i

if __name__ == "__main__":

  #import data
  #out_all_data('1110')
  s_time = 10,31,600
  e_time = 11,34,600
  out_all_data('1111-2',start_time=s_time,end_time=e_time)
