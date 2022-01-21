import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
 
from lstm_test import *
 
def lstm_predict(model_path,lstm_data_x,lstm_data_y):
   from keras.models import Sequential, load_model
   from keras.layers.core import Dense, Activation
   from keras.layers import BatchNormalization
   from keras.layers.recurrent import LSTM
   from keras.utils import np_utils
   from keras.optimizers import Adam
   
   #読み込み
   load_model = load_model(model_path)
   #load_model = load_model("./output/"+username+"_model_"+str(emo_num)+".h5")
 
   #予測
   lstm_data_y_predict = load_model.predict(lstm_data_x)
 
   plt.figure()

   plt.plot(lstm_data_y[-150:, 0], lw=2,label="data")
   plt.plot(lstm_data_y_predict[-150:, 0], '--', lw=2,label="predict")
   plt.legend()

   plt.show()

   #再帰予測
   #lstm_data_future = pd.DataFrame(index=range(300), columns=['sin', 'cos'], data=0)
   #lstm_data_future.iloc[:timesteps-1, :] = lstm_data_x[-1, :, :]
 
   #for i in lstm_data_future.index[timesteps-1:]:
   #    x = lstm_data_future.iloc[i-timesteps+1:i, :].values.reshape(1, timesteps-1, -1)
   #    y = model.predict(x)
   #    lstm_data_future.iloc[[i], :] = y
 
   #plt.figure()
   #lstm_data_future.iloc[timesteps:].plot(title='future') 

if __name__ == "__main__":
   username = '1111-2'
   number = 0

   lstm_data =[]
   lstm_data_y=[]

   #testdata maker
   #model_path="./output/"+username+"_model_"+str(number)+".h5"
   test_output_savedata = "test_resized_mood_estimated.h5"
   model_path = test_output_savedata

   lstm_data_x,lstm_data_y=lstm_mood_mkdat(username,number,lstm_data,lstm_data_y)
   x,y = reshape_dat(lstm_data_x,lstm_data_y)

   lstm_predict(model_path,x,y)

