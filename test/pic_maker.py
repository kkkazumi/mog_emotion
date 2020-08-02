import numpy as np
import cv2

filename_label = ['ad_all','imu_all','hap_out','sup_out','ang_out','sad_out','neu_out','gaze_out','head_pose_polate']
limit_error = [1000,1000,50000,50000,50000,50000,50000,50000,50000]
data_col = [8,6,3,3,3,3,3,3,3]

def min_max(x, axis=None):
  min = x.min(axis=axis, keepdims=True)
  max = x.max(axis=axis, keepdims=True)
  result = (x-min)/(max-min)
  return result

def read_files(user_name,data_type):
  data_path="/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/home/kazumi/mogura/"

  filename = data_path + user_name + '/' + filename_label[data_type]+'.csv'
  data = np.loadtxt(filename,delimiter=",")

  return data

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
    print('unit2')
    print('self.data.shape',self.data.shape)
    print('check row',self.start_row, self.end_row,data_col,self.data_type)
    unit = unit_make(self.data[self.start_row:self.end_row,:data_col[self.data_type]],shape)
    return unit

def out_all_data(username):
  mog = Data(username,0)
  imu = Data(username,1)
  hap = Data(username,2)
  sup = Data(username,3)
  ang = Data(username,4)
  sad = Data(username,5)
  neu = Data(username,6)
  gaz = Data(username,7)
  hed = Data(username,8)

  start_time = max(mog.check_start(),imu.check_start(),hap.check_start(),
  sup.check_start(),
  ang.check_start(),
  sad.check_start(),
  neu.check_start(),
  gaz.check_start(),
  hed.check_start())

  emo_data = np.loadtxt('./test_csv/emotion_test.csv',delimiter=",")
  for i in range(emo_data.shape[0]):
    end_time = emo_data[0,2],emo_data[0,3],emo_data[0,4]

    shape = 1614

    half_data = np.hstack((mog.get_unit(start_time,end_time,shape),imu.get_unit(start_time,end_time,shape),
    hap.get_unit(start_time,end_time,shape),
    sup.get_unit(start_time,end_time,shape),
    ang.get_unit(start_time,end_time,shape),
    sad.get_unit(start_time,end_time,shape),
    neu.get_unit(start_time,end_time,shape),
    gaz.get_unit(start_time,end_time,shape),
    hed.get_unit(start_time,end_time,shape)))

    #np.savetxt(username+'test_class_'+str(i)+'.csv',half_data,delimiter=",")
    cv2.imwrite('data_test.png',half_data.T)


if __name__ == "__main__":

  #import data
  out_all_data('1107-1')
