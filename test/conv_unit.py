from sklearn.decomposition import PCA
import optuna

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import scipy as sp
from scipy.stats import pearsonr


#name list must be int number
name_list = [1]
mogdir_path = '/media/kazumi/4b35d6ed-76bb-41f1-9d5d-197a4ff1a6ab/home/kazumi/mogura/'

INPUT_DIM = input_dim
OUTPUT_DIM = output_dim
EPOCH = 50000

for name_num in name_list:
  dir_name = mogdir_path + str(name_num+1)
  for set_num in (5,10,15,20,25,30,35,40):

    r = np.zeros(30)
    p = np.zeros(30)
    for i in range(30):#calculate the avelage value of estimated accuracy for datum in this loop
      i_csv = str(set_num) + "-" + str(i) + ".csv"

      #set input, output data to train
      factor_train = dir_name + "/factor_train" + i_csv
      mood_train = dir_name + "/mood_train" + i_csv
      face_train = dir_name + "/face_train" + i_csv

      x1_train = np.loadtxt(factor_train,delimiter=',')
      x2_train = np.loadtxt(mood_train,delimiter=',')
      x_train = np.hstack((x1_train,x2_train.reshape(-1,1)))

      y_train = np.loadtxt(face_train,delimiter=',')

      def objective(trial):
        #セッションのクリア
        K.clear_session()

        #最適化するパラメータの設定
        #中間層のユニット数
        mid_units = trial.suggest_int("mid_units", 5, 100)

        #活性化関数
        activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])

        #optimizer
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", "rmsprop"])

        #define NN framework
        model = Sequential()
        model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
        model.add(Dense(OUTPUT_DIM, activation=activation))

        model.compile(optimizer=optimizer,
              loss="mean_squared_error",
              metrics=["accuracy"])

        history = model.fit(x_train, y_train, verbose=0, epochs=200, batch_size=128, validation_split=0.1)
        return 1 - history.history["val_acc"][-1]

      study = optuna.create_study()
      study.optimize(objective, n_trials=100)

      mid_units=study.best_params["mid_units"]
      activation=study.best_params["activation"]
      optimizer=study.best_params["optimizer"]

      #define NN framework
      model = Sequential()
      model.add(Dense(mid_units, input_dim = INPUT_DIM, activation=activation))
      model.add(Dense(OUTPUT_DIM, activation=activation))

      model.compile(optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["accuracy"])

      train=model.fit(x=x_train, y=y_train, nb_epoch=EPOCH)

      #test to output
      factor_test = np.loadtxt(dir_name + "/factor_test" + i_csv ,delimiter=',')
      mood_test = np.loadtxt(dir_name + "/mood_test" + i_csv ,delimiter=',')
      face_test = np.loadtxt(dir_name + "/face_test" + i_csv ,delimiter=',')

      m_pred = np.zeros_like(mood_test)

      #set data to test
      for m_t in range(len(mood_test)):
        x_tes = np.tile(factor_test[m_t],(100,1))
        m_candidate = np.linspace(0,1,100)
        y_ans = np.tile(face_test[m_t],(100,1))

        xtest = np.hstack((x_tes,m_candidate.reshape(-1,1)))
        ytest = model.predict(xtest)
        err = ytest - y_ans
        err_array = np.sum(err,1)
        m_est = np.argmin(abs(err_array))
        m_pred[m_t] = m_est/100.0

      #pearson r
      r[i], p[i] = pearsonr(mood_test, m_pred)

      #test output
      outname = dir_name + "/estimated_nn" + i_csv
      np.savetxt(outname,m_pred,fmt="%.3f")
      lossname = dir_name + "/nn_loss" + i_csv
      np.savetxt(lossname,train.history['loss'])
    np.savetxt(dir_name+"/nn_corr-"+str(set_num)+".csv",r,fmt='%.5f',delimiter=',')
    np.savetxt(dir_name+"/nn_p-"+str(set_num)+".csv",p,fmt='%.5f',delimiter=',')
