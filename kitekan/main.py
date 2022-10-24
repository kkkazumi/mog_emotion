import numpy as np
from get_factor import *

MENTAL = "mental"
EMOTION = "emotion"
SIGNAL = "signal"

def tmp_func_f(weight,m):
  m_new = m+weight*1
  return m_new

def get_mental(t,weights,m_before=0):
  if(t==0):
    mental=5
  else:
    mental=tmp_func_f(weights,m_before)
  return mental

def get_weight(func_type):
  if(func_type==MENTAL):
    weights=1
  elif(func_type==EMOTION):
    weights=0
  elif(func_type==SIGNAL):
    weights=0
  return weights

def get_emotion(t):
  #tmp emotion
  emotion=0
  return emotion

def main():
  t=0
  m=[]
  w_f=get_weight(MENTAL)
  m.append(get_mental(t,w_f))
  factor=dummy_factors()
  print(factor)
  while(t<50):
    t+=1
    m.append(get_mental(t,w_f,m_before=m[-1]))
    #if(t>50):

if __name__ == '__main__':
  main()
