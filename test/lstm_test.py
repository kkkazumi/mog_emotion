import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.datasets import load_iris

timesteps = 100
 
def lstm_mkdat(username,number,lstm_data,lstm_data_y):

  data_x0 = np.loadtxt('./output/'+username+'_face_test_class_'+str(number)+'.csv',delimiter=",")
  data_z0 = np.zeros((data_x0.shape[0],1,1))
  data_z0[-1,:,:] = 1#emo_data[0,1]

  data_x = data_x0
  data_z = data_z0

  length=data_x.shape[0]
  data_dim = data_x.shape[1]
  print("data_dim",data_dim)

  #lstm_data = []
  #lstm_data_y = []

#for i in range(100): #for repeat test
  #for t in range(5):
  for i in range(length-2*timesteps,length-timesteps):
    lstm_data.append(data_x[i:i+timesteps])
    lstm_data_y.append(data_z[i+timesteps])
  print("last i ",i)

  return lstm_data, lstm_data_y

def reshape_dat(lstm_data,lstm_data_y):
  lstm_data_x = np.array(lstm_data)
  lstm_data_x0 = np.array(lstm_data)
  lstm_data_y = np.array(lstm_data_y)
  lstm_data_x = lstm_data_x.reshape(lstm_data_x.shape[0],timesteps,-1)
  lstm_data_y = lstm_data_y.reshape(lstm_data_y.shape[0],-1)

  return lstm_data_x, lstm_data_y

def lstm_learn(lstm_data_x,lstm_data_y,data_name):
  data_dim = lstm_data_x.shape[2]

  from keras.models import Sequential, load_model
  from keras.layers.core import Dense, Activation
  from keras.layers import BatchNormalization
  from keras.layers import Conv2D, MaxPooling2D
  from keras.layers.recurrent import LSTM
  from keras.utils import np_utils
  from keras.optimizers import Adam

  #モデル定義
  hidden = 100
  model = Sequential()
  model.add(LSTM(hidden, input_shape=(timesteps,data_dim), stateful=False, return_sequences=False))
  model.add(Dense(lstm_data_y.shape[1]))
  model.compile(loss="mean_squared_error", optimizer='adam')

  #学習
  model.fit(lstm_data_x, lstm_data_y,
           batch_size=32,
           epochs=1,
           validation_split=0.1,
           )
  print("ok")

  #保存と読み込み
  model.save(data_name)
  #load_model = load_model("./output/sincos_model.h5")

  '''
  #予測
  lstm_data_y_predict = model.predict(lstm_data_x)

  plt.figure()

  plt.plot(lstm_data_y[-150:-50, 0], lw=2,label="data")
  plt.plot(lstm_data_y_predict[-150:-50, 0], '--', lw=2,label="predict")
  plt.legend()

  plt.show()
  '''

if __name__ == "__main__":

  filename = '1107-1'
  
  for i in range(3):
    lstm_data,lstm_data_y=lstm_mkdat(filename,i)
    lstm_data_x, lstm_data_y=reshape_dat(lstm_data,lstm_data_y)
    data_name = './output/'+filename+'_model_'+str(i)+'.h5'
    lstm_learn(lstm_data_x,lstm_data_y,data_name)
