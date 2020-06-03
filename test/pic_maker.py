import numpy as np
import cv2

def min_max(x, axis=None):
  min = x.min(axis=axis, keepdims=True)
  max = x.max(axis=axis, keepdims=True)
  result = (x-min)/(max-min)
  return result

def unit_make(data, data_len):
  tsize=data.shape[0]
  ysize=data.shape[1]

  comp_data = cv2.resize(data, (ysize,data_len))
  return comp_data

def get_time(data_array,target_row):
  data_min = data_array[target_row,-3]
  data_sec = data_array[target_row,-2]
  data_msec = data_array[target_row,-1]
  target_time = data_min,data_sec,data_msec
  return target_time

def check_time(check_array,target_time):
  target_min, target_sec, target_msec = target_time
  data=np.where((check_array[:,-3]==target_min)&(check_array[:,-2]==target_sec)&((check_array[:,-1]>target_msec-1000)&(check_array[:,-1]<target_msec+1000)))
  _mostmin=np.argmin(abs(check_array[data,-1]-target_msec))
  same_time_row=data[0][_mostmin]
  
  return same_time_row
  

if __name__ == "__main__":

  #imu_data = np.loadtxt('test_imu.csv',delimiter=",")
  emo_data = np.loadtxt('emotion_test.csv',delimiter=",")
  mogura_data = np.loadtxt('test_mogura_all.csv',delimiter=",")
  imu_data = np.loadtxt('imu_test_2.csv',delimiter=",")
  

  #std_st_row = 3129
  #std_en_row = 3798
  std_st_row = 0
  end_time = emo_data[0,2],emo_data[0,3],emo_data[0,4]

  std_en_row = check_time(imu_data,end_time)
  shape = std_en_row-std_st_row
  start_time_a = get_time(imu_data,std_st_row)
  start_time_b = get_time(mogura_data,std_st_row)
  start_time = max(start_time_a,start_time_b)
  print(start_time)

  #end_time = get_time(imu_data,std_en_row)
  st=check_time(mogura_data,start_time)
  en=check_time(mogura_data,end_time)
  print(st,en)

  trim_mog= unit_make(mogura_data[st:en,:8],shape)
  imu = imu_data[std_st_row:std_en_row,:6]
  half_data=np.hstack((trim_mog,imu))

  CHECK_ROW = 2

  half_data = min_max(half_data)
  np.savetxt("data.csv",half_data,delimiter=",")
  cv2.imwrite('data.png',(half_data*255).T)

  #from matplotlib import pyplot as plt
  #for i in range(8):
  #  plt.plot(trim_mog[:,i])
  #  plt.show()
