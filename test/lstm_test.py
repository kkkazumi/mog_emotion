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

  username = '1107-1'
  i=0

  """
  data_x1 = np.loadtxt(username+'test_class_'+str(i)+'.csv',delimiter=",")
  emo_data = np.loadtxt('./test_csv/emotion_test.csv',delimiter=",")

  for i in range(emo_data.shape[0]):
    data_x1 = np.loadtxt(username+'test_class_'+str(i)+'.csv',delimiter=",")
    data_z1 = np.zeros((data_x1.shape[0],1,1))
    data_z1[-1,:,:] = 1#emo_data[0,1]
  """

  data_x0 = np.loadtxt('1107-1test_class_0.csv',delimiter=",")
  data_z0 = np.zeros((data_x0.shape[0],1,1))
  data_z0[-1,:,:] = 1#emo_data[0,1]

  #data_x1[:data_x0.shape[0],:] = data_x0
  #data_z1[:data_x0.shape[0],:] = data_z0

  data_x = data_x0
  data_z = data_z0

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
  lstm_data_x0 = np.array(lstm_data)
  lstm_data_y = np.array(lstm_data_y)
  lstm_data_x = lstm_data_x.reshape(lstm_data_x.shape[0],timesteps,-1)
  lstm_data_y = lstm_data_y.reshape(lstm_data_y.shape[0],-1)

  print(lstm_data_x.shape)
  print(lstm_data_y.shape)

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

  """
  model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(timesteps,data_dim,1)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dense(10,activation='softmax'))
  model.add(LSTM(hidden, input_shape=(10,), stateful=False, return_sequences=False))
  """
  model.add(LSTM(hidden, input_shape=(timesteps,data_dim), stateful=False, return_sequences=False))
  model.add(Dense(lstm_data_y.shape[1]))
  model.compile(loss="mean_squared_error", optimizer='adam')

#学習
  model.fit(lstm_data_x, lstm_data_y,
           batch_size=32,
           epochs=100,
           validation_split=0.1,
           )
  print("ok")

#保存と読み込み
  model.save("data_comb_test_model.h5")
  load_model = load_model("sincos_model.h5")

#予測
  lstm_data_y_predict = model.predict(lstm_data_x)

  plt.figure()

  plt.plot(lstm_data_y[-150:-50, 0], lw=2,label="data")
  plt.plot(lstm_data_y_predict[-150:-50, 0], '--', lw=2,label="predict")
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
