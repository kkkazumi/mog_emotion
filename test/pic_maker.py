import numpy as np
import cv2

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
  mogura_data = np.loadtxt('test_mogura_all.csv',delimiter=",")
  imu_data = np.loadtxt('test_imu.csv',delimiter=",")

  std_st_row = 3129
  std_en_row = 3799
  shape = std_en_row-std_st_row

  start_time = get_time(imu_data,std_st_row)
  end_time = get_time(imu_data,std_en_row)
  st=check_time(mogura_data,start_time)
  en=check_time(mogura_data,end_time)
  print(st,en)

  trim_mog= unit_make(mogura_data[st:en,:8],shape)
  imu = imu_data[std_st_row:std_en_row,:6]
  half_data=np.hstack((trim_mog,imu))

  CHECK_ROW = 2

  cv2.imwrite('data.png',(half_data*255).T)

  from matplotlib import pyplot as plt
  for i in range(8):
    plt.plot(trim_mog[:,i])
    plt.show()
