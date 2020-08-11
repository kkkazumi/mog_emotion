import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
 
from sklearn.datasets import load_iris
 
def sin(T=500):
   x = np.arange(0, 2*T+1)
   y = np.arange(0, 2*T+1)
   sin = np.sin(2.0 * np.pi * x / T).reshape(-1, 1)+np.sin(2.0 * np.pi * y / T).reshape(-1, 1)

   return x,y,sin

def ret_sin(x):
   T=500
   sin = np.sin(2.0 * np.pi * x / T).reshape(-1, 1)
   return sin
 
if __name__ == "__main__":
 
   username = '1110'
   #データセット作り
   #data_x, data_y ,data_z = sin()
   data_x = np.loadtxt('./output/face_test_class_12.csv',delimiter=",")
   #emo_data = np.loadtxt('../test_csv/emotion_test.csv',delimiter=",")
   df = pd.read_csv('../emo_questionnaire/'+username+'.csv',header=None)#,delimiter=",",dtype="unicode")
   emo_data = np.zeros((df.shape[0],df.shape[1]-1),dtype=np.int)
   emo_data = df.values[:,:5]
   print(data_x.shape[0])
   data_z = np.zeros((data_x.shape[0],1,1))
   print(emo_data[0,1])
   data_z[-1,:,:] = 1#emo_data[0,1]

   timesteps = 100
   length=data_x.shape[0]
   data_dim = data_x.shape[1]
   print("data_dim",data_dim)
 
   lstm_data = []
   lstm_data_y = []
   index_data = []
   index_data_y = []

   #for i in range(100): #for repeat test
   for t in range(10):
     for i in range(length-2*timesteps,length-timesteps):
         lstm_data.append(data_x[i:i+timesteps])
         lstm_data_y.append(data_z[i+timesteps])
     print("last i ",i)
   #i = 84477

   lstm_data_x = np.array(lstm_data)
   lstm_data_y = np.array(lstm_data_y)
   lstm_data_x = lstm_data_x.reshape(lstm_data_x.shape[0],timesteps,-1)
   lstm_data_y = lstm_data_y.reshape(lstm_data_y.shape[0],-1)

   from keras.models import Sequential, load_model
   from keras.layers.core import Dense, Activation
   from keras.layers import BatchNormalization
   from keras.layers.recurrent import LSTM
   from keras.utils import np_utils
   from keras.optimizers import Adam
   
   #モデル定義
   #hidden = 100
   #model = Sequential()
   #model.add(LSTM(hidden, input_shape=(timesteps, data_dim), stateful=False, return_sequences=False))
   #model.add(Dense(lstm_data_y.shape[1]))
   #model.compile(loss="mean_squared_error", optimizer='adam')
 
   #保存と読み込み
   load_model = load_model("./output/data_comb_test_model.h5")
 
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
