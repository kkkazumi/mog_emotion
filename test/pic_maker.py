import numpy as np
import cv2
import pandas as pd

import os
import matplotlib.pyplot as plt

from get_mood_data import *

filename_label = ['ad_all','imu_all','hap_out','sup_out','ang_out',
                  'sad_out','neu_out','gaze_out','head_pose_pl1','head_pose_pl2',
                  'head_pose_pl3']#11
limit_error = [1000,1000,50000,50000,50000,
              50000,50000,50000,50000,100000,100000,
              100000]#11
data_col = [8,6,3,3,3,
            3,3,3,3,3,
            3]#11

from sklearn.preprocessing import MinMaxScaler

def nomalize(xx,zz):
#hap,sup,ang,sad,neu, ad(ph1,ph2,ph3,ph4,ph5,ph6,ph7,hammer),imu=data
  min_li = [0,0,0,0,0,2.7,0,0,0,0,0,0,0,0,-20,-20,-20,-250,-2500,-1000]
  max_li = [100,100,100,100,100,3.5,0.5,0.6,0.4,0.5,0.6,0.8,3.5,20,20,20,250,2500,1000]

  mnscaler = MinMaxScaler(feature_range=(0,255),copy=True)
  mnscaler.fit(np.hstack((xx,np.reshape(zz,(xx.shape[0],-1)))))
  x = mnscaler.transform()
  return x,z

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
  #print("unit make data",data.shape)
  tsize=data.shape[0]
  ysize=data.shape[1]
  #data_len = data.shape[0]
  comp_data = cv2.resize(data, (ysize,data_len))
  print("data compress:",tsize,"to",data_len)
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
  print("in check time and data shape",data)
  _mostmin=np.argmin(abs(check_array[data,-1]-target_msec))
  same_time_row=data[0][_mostmin]
  #print("check time",same_time_row)
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

  def get_size(self,start_time,end_time):
    self.set_start_data(start_time,end_time)
    data_size = self.end_row - self.start_row
    return data_size

  def get_unit(self,start_time,end_time,shape):
    self.set_start_data(start_time,end_time)
    print("start row,end row",self.start_row,self.end_row,data_col[self.data_type])
    unit = unit_make(self.data[self.start_row:self.end_row,:data_col[self.data_type]],shape)
    print("got unit",filename_label[self.data_type])
    return unit

def average(data,size):
  #import matplotlib.pyplot as plt
  b = np.ones(size)/size
  moving_average = np.zeros_like(data)
  for i in range(data.shape[1]):
    moving_average[:,i] = np.convolve(data[:,i],b,mode='same')
  return moving_average

def data2file(data,start_time,end_time,filename,str_part):
    hap,sup,ang,sad,neu,ad,imu,gz,hd1,hd2,hd3=data
    #print('check shape',shape)
    shape = min(hap.get_size(start_time,end_time),
      sup.get_size(start_time,end_time),
      ang.get_size(start_time,end_time),
      sad.get_size(start_time,end_time),
      neu.get_size(start_time,end_time),
      ad.get_size(start_time,end_time),
      imu.get_size(start_time,end_time),
      gz.get_size(start_time,end_time),
      hd1.get_size(start_time,end_time),
      hd2.get_size(start_time,end_time),
      hd3.get_size(start_time,end_time))
    print("shape",shape)
    input()

    ave_size = 5
    
    half_data = np.hstack((hap.get_unit(start_time,end_time,shape),
    sup.get_unit(start_time,end_time,shape),
    ang.get_unit(start_time,end_time,shape),
    sad.get_unit(start_time,end_time,shape),
    neu.get_unit(start_time,end_time,shape),
    ad.get_unit(start_time,end_time,shape),
    imu.get_unit(start_time,end_time,shape),
    gz.get_unit(start_time,end_time,shape),
    hd1.get_unit(start_time,end_time,shape),
    hd2.get_unit(start_time,end_time,shape),
    hd3.get_unit(start_time,end_time,shape)))

    print("end all of get unit")
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

  ad = Data(username,0)#the 2nd arg is data type (refer #filename_label)
  imu = Data(username,1)#the 2nd arg is data type (refer #filename_label)

  gz = Data(username,7)
  hd1 = Data(username,8)
  hd2 = Data(username,9)
  hd3 = Data(username,10)


  #TODO: add gaze and headpose data..

  if(start_time == None):
    start_time = max(hap.check_start(),
    sup.check_start(),
    ang.check_start(),
    sad.check_start(),
    neu.check_start(),
    ad.check_start(),
    imu.check_start(),
    gz.check_start(),
    hd1.check_start(),
    hd2.check_start(),
    hd3.check_start())

  print("start time",start_time)
  print("end time",end_time)

  qfile_path = '../emo_questionnaire/'+username+'.csv'
  i=0
  data = hap,sup,ang,sad,neu,ad,imu,gz,hd1,hd2,hd3

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
      yy = np.array(cv2.resize(y,dsize=(1,mag)))
      yline=cv2.flip(yy, 0)
      resized_mood = yline[:,0]

      np.savetxt("test_resized_mood.csv",resized_mood,delimiter=",")

    #data2file(data,end_time,add1sec(end_time),filename,'2nd')

    return i

if __name__ == "__main__":

  #import data
  #out_all_data('1110')
  s_time = 10,31,600000
  e_time = 11,3,100000
  out_all_data('1111-2',start_time=s_time,end_time=e_time)
