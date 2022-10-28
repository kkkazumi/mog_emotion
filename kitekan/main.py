import numpy as np
from get_factor import *

MENTAL = "mental"
EMOTION = "emotion"
SIGNAL = "signal"

STEP = 0.1

def tmp_func_f(weight,f):
  m_new = np.dot(f,weight)
  return m_new

def func_g(weight)

def func_h(weight)

def argmin_ndim(arg_array):
  return np.unravel_index(arg_array.argmin(), arg_array.shape)

def get_dummy_weight(func_type):
  if(func_type==MENTAL):
    weights=(np.random.randint(0,10,5)-5*np.ones(5))/10.0
  elif(func_type==EMOTION):
    weights=(np.random.randint(0,10,5)-5*np.ones(5))/10.0
  elif(func_type==SIGNAL):
    weights=(np.random.randint(0,10,5)-5*np.ones(5))/10.0
  return weights

def est_emotion(func_type,weights):
  if(func_type==EMOTION):
    #g()
    emotion=func_g(weights)
  elif(func_type==SIGNAL):
    #h()
    emotion=func_h(weights)
  return emotion

def est_mental(t,weights,factor=0,m_before=0,length="30",mode="dummy"):
  if(mode=="dummy"):
    if(t==0):
      mental=np.array([5])
    else:
      mental=np.array([m_before])
      for i in range(1,length):
        _m=mental[i-1]+tmp_func_f(weights,factor[i])
        mental=np.append(mental,_m)
  else:
    mental=np.array([5])
  return mental

def calc_emo_err(weights,args):
  w_f,w_g,w_h=weights
  factor,mental,signal=args

  emotion_g = est_emotion(EMOTION,w_g,factor,mental)
  emotion_h = est_emotion(SIGNAL,w_h,factor,mental,signal)
  err = abs(emotion_g - emotion_h)
  return err

def fit_weight(func_type,arg=0,step=0,mode="dummy"):
  if(step is not 0):
    if(func_type == MENTAL):
      arg = factor,mental
    elif(func_type == EMOTION):
      arg = factor,mental,emotion
    elif(func_type == SIGNAL):
      arg = factor,mental,signal

  if(mode=="dummy"):
    weights=get_dummy_weight(func_type)
  else:
    if(func_type==MENTAL):
      #weights = np.loadtxt(weight_file)
      weights=0
    elif(func_type==EMOTION):
      weights=0
    elif(func_type==SIGNAL):
      weights=0
  return weights

def get_emotion(t,func_type,mode="dummy"):
  #tmp emotion
  if(func_type==EMOTION):
    emotion=0
  elif(func_type==SIGNAL):
    emotion=0
  return emotion

def get_factor_data(mode="dummy"):
  if(mode=="dummy"):
    factor=dummy_factors()
  else:
    factor=0
  return factor

def get_signal_data(mode=="dummy"):
  if(mode=="dummy"):
    signals=dummy_signals()
  else:
    signals=0
  return signals

def find_mental_grad(step,weights,args):
  factor, mental, signal = args

  if(len(mental)>1):
    time_length = len(mental[0])
    dim=len(mental)
  else:
    time_length=len(mental)
    mental = np.reshape(mental,(-1,time_length))
    dim=1
  err_array = np.zeros((dim,time_length))
  for t in range(time_length):
    for i in range(dim):
      mental[i,t] += step
      args = factor,mental,signal
      err=calc_emo_err(weights,args)
      err_array[i,t] = err

      mental[i,t] -= step #reset

  min_index = argmin_ndim(err_array)
  new_mental = np.copy(mental)
  new_mental[min_index] = mental[min_index]+step
#TODO: define necessary functions for below
  emotion = 
  new_weights = 

  return new_weights,new_mental,emotion

def search(weights,args):
  w_f,w_g,w_h = weights
  factor,mental,signal = args
  err=1000#is it enough?

  while(err>10):

    #search emotion and mental
    new_weights,new_mental,emotion = find_mental_grad(STEP,weights,args)

  return new_weights,new_mental,emotion

def main(length,mode):
  t=0
  w_f=fit_weight(MENTAL,step=0)
  w_g=fit_weight(EMOTION,step=0)
  w_h=fit_weight(SIGNAL,step=0)

  mental=est_mental(t,w_f,mode=mode)
  factor=get_factor_data(mode)
  signal=get_signal_data(mode)

  j=0
  while(t<len(factor)):
    t+=1
    j+=1
    if(t>length):
      if(j>length/2):

        add_m=est_mental(t,w_f,f,mental[-1],length=length,mode=mode)
        mental=np.append(mental,add_m)

        m=mental[t-length:t]
        f=factor[t-length:t]
        s=signal[t-length:t]

        w_f,w_g,w_h = weights_search
        f,m,s=args_search

        #update values
        new_weights,mental,emotion = search(weights_search,args_search)
        new_weights = w_f,w_g,w_h

        #重みを更新して、と思ったけど、手順違った？
        #arg_h = f,mental,s
        #w_h=fit_weight(SIGNAL,arg=arg_h)

        #arg_g = f,mental,emotion
        #w_g=fit_weight(EMOTION,arg=arg_g)

        j=0
  print(mental)

if __name__ == '__main__':
  length=30
  mode="dummy"
  main(length,mode)
