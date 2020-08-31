from pic_maker import *
from lstm_test import *
from only_test import *

import pathlib
import os

dir_path = "/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/home/kazumi/mogura/"
p_temp = pathlib.Path(dir_path).glob('*11*')
emo_len = np.zeros(19)
file_list = list(p_temp)

#at beginning, make all data picture for learning.
for file_path in sorted(file_list):
  filename = os.path.splitext(os.path.basename(file_path))[0]
  i = file_list.index(file_path)
  emo_len[i] = out_all_data(filename)

lstm_data = []
lstm_data_y = []

for target_file_path in sorted(file_list):
  target_filename = os.path.splitext(os.path.basename(target_file_path))[0]
  target_i = file_list.index(target_file_path)
  for target_emo_num in range(emo_len(target_i)):

    for file_path in sorted(file_list):
      filename = os.path.splitext(os.path.basename(file_path))[0]
      i = file_list.index(file_path)
      for emo_num in range(emo_len(i)):
        data_name = './output/'+filename+'_model_'+str(emo_num)+'.h5'

        if filename is not target_filename:
          lstm_data,lstm_data_y=lstm_mkdat(emo_num,lstm_data,lstm_data_y)
        if filename == target_filename:
          if emo_num is not target_emo_num:
            lstm_data_x, lstm_data_y=reshape_dat(lstm_data,lstm_data_y)
            lstm_learn(lstm_data_x,lstm_data_y,data_name)
          elif emo_num == target_emo_num:
            print('skip target',target_filename, target_emo_num)

    lstm_data_x,lstm_data_y=lstm_mkdat(filename,target_emo_num)
    data_name = './output/'+filename+'_model_'+str(emo_num)+'.h5'
    lstm_data_x,lstm_data_y = lstm_learn(lstm_data_x,lstm_data_y,data_name)
    lstm_predict(data_name,lstm_data_x,lstm_data_y)

#if __name__ == "__main__":
