import numpy as np
from get_factor import *

MENTAL = "mental"
EMOTION = "emotion"
SIGNAL = "signal"

def tmp_func_f(weight,f):
  m_new = np.dot(f,weight)
  return m_new

def get_mental(t,weights,factor=0,m_before=0,length="30",mode="dummy"):
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

def get_weight(func_type,mode="dummy"):
  if(mode=="dummy"):
    if(func_type==MENTAL):
      weights=(np.random.randint(0,10,5)-5*np.ones(5))/10.0
    elif(func_type==EMOTION):
      weights=0
    elif(func_type==SIGNAL):
      weights=0
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

def get_factors(mode="dummy"):
  if(mode=="dummy"):
    factor=dummy_factors()
  else:
    factor=0
  return factor

def main(length,mode):
  t=0
  w_f=get_weight(MENTAL)
  mental=get_mental(t,w_f,mode=mode)
  factor=get_factors(mode)

  j=0
  while(t<len(factor)):
    t+=1
    j+=1
    if(t>length):
      if(j>length/2):
        f=factor[t-length:t]
#TODO:1, get signal(dummy mode)
        add_m=get_mental(t,w_f,f,mental[-1],length=length,mode=mode)
        mental=np.append(mental,add_m)

#TODO:2, get weights EMOTION,SIGNAL
#TODO:3, get emotion by EMOTION
#TODO:4, get emotion by SIGNAL

        j=0
  print(mental)

if __name__ == '__main__':
  length=30
  mode="dummy"
  main(length,mode)
