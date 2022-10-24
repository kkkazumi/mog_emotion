import numpy as np
from func import *

func_num = [[0,1,2,3],[4,5,6,7],[8,9,10,11],
  [12,13,14,15],[16,17,18,19],[20,21,22,23],
  [24,25,26,27],[28,29,30,31],[32,33,34,35],[36,37,38,39]]

norm_val = [80,80,80,80,
            1,1,1,1,
            1,1,1,1,
            1,1,1,1,
            1,1,1,1,
            1,1,1,1,
            1,1,1,1,
            320,320,320,320,
            80,80,80,80,
            80,80,80,80]

def inv_norm(num, val):
  if num == 28 or num == 29 or num==30 or num==31:
    val = val * norm_val[num] - 80.0
  else:
    val = val*norm_val[num]
  return val

def sig(factor,a,b,c):
  ret=0.0
  ret = 1.0/(1.0+np.exp(a*(b*factor-c)))
  return ret

def gauss(factor,a,b,c):
  ret=0.0
  ret = np.exp(-1.0*np.power((a*factor-b),2.0)/c)
  return ret

def inv_down(factor,a,b):
  ret=0.0
  A = np.power(2.0,a)
  B = A - 1.0
  C = b * factor + 1
  D = np.power(C,a)
  ret = A/(B*D) - 1.0/B
  return ret

def inv_up(factor,a,b):
  ret = 0.0
  A = np.power(2.0,a)
  B = A - 1.0
  C = b * factor + 1
  D = np.power(C,a)
  ret = (D-1.0)/B
  return ret

func_name_str = ["time of trial hap","time of trial sup", "time of trial ang", "time of trial sad",
  "rate of wins hap","rate of wins sup","rate of wins ang","rate of wins sad",
  "encourage hap","encourage sup","encourage ang","encourage sad",
  "symp hap","symp sup","symp ang","symp sad",
  "teas hap","teas sup","teas ang","teas sad",
  "un related hap","un related sup","un related ang","un related sad",
  "no action hap","no action sup","no action ang","no action sad",
  "total point hap","total point sup","total point ang","total point sad",
  "cons win hap","cons win sup","cons win ang","cons win sad",
  "cons lose hap","cons lose sup","cons lose ang","cons lose sad"]


#//1〜10種類ある予測関数番号をもらうと計算をする関数を作ります
def func(factor, mental, func_num):
  ### time of trial ##
  if func_num == 0:
    #time of trial happy
    a = 2.0*(mental+3.0)
    b = 1.0/80.0
    c = 0.3
    ret = sig(factor,a,b,c)
  if func_num == 1:
    #time of trial sup
    a = 2.0*(mental+5.0)
    b = 1.0/80.0
    c = 0.5
    ret = sig(factor,a,b,c)
  if func_num == 2:
    #time of trial ang
    a = -3.0*(mental + 3.0)
    b = 1.0 / 80.0
    c = 0.3
    ret = sig(factor,a,b,c)
  if func_num == 3:
    #time of trial sad
    a = 1.0/80.0
    b = 0.3
    c = (11.0-mental)/300.0
    ret = gauss(factor,a,b,c)

  ### rate of wins
  if func_num == 4:
    # rate of wins happy
    a = 1.1*(6.0-mental/3.0)
    b = 1.0
    ret = inv_up(factor,a,b)
  if func_num == 5:
    # rate of wins sup
    a = 11.0-mental
    b = 1.0
    ret = inv_up(factor,a,b)

  if func_num == 6:
    #rate of wins ang
    a = 2.0*(mental + 5.0)
    b = 1.0
    c = 0.5
    ret = sig(factor,a,b,c)

  if func_num == 7:
    # rate of wins sad
    a = 2.0*(mental/2.0+4.0)
    b = 1.0
    ret = inv_down(factor,a,b)

		#ret= np.exp(-(factor + 0.3) / (11 - mental)*7.0)#//正答率sad

  if func_num == 8:
    #encourage hap
    a = 1.0
    b = 0.7
    c = mental/100.0
    ret = gauss(factor,a,b,c)
  if func_num == 9:
    #encourage sup
    a = 1.0
    b = 0.8
    c = mental / 80.0
    ret = gauss(factor,a,b,c)
  if func_num == 10:
    #encourage ang
    a = 0.9 * mental
    b = 1.0
    ret = inv_down(factor,a,b)
  if func_num == 11:
    #encourage sad
    a = mental
    b = 1.0
    ret = inv_down(factor,a,b)

  if func_num == 12:
    #symp hap
    a = 1.0
    b = 0.4
    c = mental / 300.0
    ret = gauss(factor,a,b,c)
  if func_num == 13:
    #symp sup
    a = 1.0
    b = 0.6
    c = mental / 300.0
    ret = gauss(factor,a,b,c)
  if func_num == 14:
    #symp ang
    a = 0.9 * mental
    b = 1.0
    ret = inv_down(factor,a,b)
  if func_num == 15:
    #symp sad
    a = mental
    b = 1.0
    ret = inv_down(factor,a,b)

  if func_num == 16:
    #teasing hap
    a = 1.0
    b = 0.2
    c = mental/300.0
    ret = gauss(factor,a,b,c)
  if func_num == 17:
    #teasing sup
    a = 1.0
    b = 0.5
    c = mental/200.0
    ret = gauss(factor,a,b,c)
  if func_num == 18:
    #teasing ang
    a = 0.8 * mental
    b = 1.0
    ret = inv_up(factor,a,b)
  if func_num == 19:
    #teasing sad
    a = 0.8 * mental
    b = 1.0
    ret = inv_up(factor,a,b)

  if func_num == 20:
    #unrelated hap
    a = 1.0
    b = 0.6
    c = mental/100.0
    ret = gauss(factor,a,b,c)
  if func_num == 21:
    #unrelated sup
    a = 11.0 - mental
    b = 1.0
    ret = inv_up(factor,a,b)
  if func_num == 22:
    #unrelated ang
    a = 0.9 * mental
    b = 1.0
    ret = inv_up(factor,a,b)
  if func_num == 23:
    #unrelated sad
    a = 1.4 * mental
    b = 1.0
    ret = inv_up(factor,a,b)

  if func_num == 24:
    #no action hap
    a = 11.0 - mental
    b = 1.0
    ret = inv_down(factor,a,b)
  if func_num == 25:
    #no action sup
    a = 11.0 - mental
    b = 1.0
    ret = inv_up(factor,a,b)
  if func_num == 26:
    #no action ang
    a = 0.7 * (mental/2.0)+1.0
    b = 1.0
    ret = inv_up(factor,a,b)
  if func_num == 27:
    #no action sasd
    a = 2.0 * mental + 1.0
    b = 1.0
    ret = inv_up(factor,a,b)

  if func_num == 28:
    #total point hap
    a = 14.0 * (mental + 2.0)
    b = 1.0 / 240.0
    c = 0.06 * (5.5 - mental / 2.0)
    ret = sig(factor,a,b,c)
  if func_num == 29:
    #total point sup
    a = 2.0 * (2.0 * mental + 5.0)
    b = 1.0 / 240.0
    c = 0.06 * (6.5 - mental / 2.0)
    ret = sig(factor,a,b,c)
  if func_num == 30:
    #total point ang
    a = -4.0 * (1.5 * mental + 4.0)
    b = 1.0 / 240.0
    c = 0.01 * (7.0 - 2.0 * mental)
    #print("total point c",c)
    ret = sig(factor,a,b,c)
  if func_num == 31:
    #total point sad
    a = -3.0 * (1.5 * mental + 4.0)
    b = 1.0 / 240.0
    c = 0.01 * (5.0 - 3.0 * mental)
    ret = sig(factor,a,b,c)
  if func_num == 32:
    #consecutive wins hap
    a = -50.0 * (mental / 11.0 + 0.5)
    b = 1.0 / 80.0
    c = 0.1 * (3.0 - 0.2 * mental)
    ret = sig(factor,a,b,c)
  if func_num == 33:
    #consecutive wins sup
    a = -70.0 * (mental / 11.0 + 0.5)
    b = 1.0 / 80.0
    c = 0.15 * (3.0 - 0.2 * mental)
    ret = sig(factor,a,b,c)
  if func_num == 34:
    #consecutive wins ang
    a = 0.6 * mental
    b = 1.0 / 80.0
    ret = inv_down(factor,a,b)
  if func_num == 35:
    #consecutive wins sad
    a = 2.0 * mental
    b = 1.0 / 80.0
    ret = inv_down(factor,a,b)

  if func_num == 36:
    #consecutive losses hap
    a = 0.7 * mental
    b = 1.0 / 80.0
    ret = inv_down(factor,a,b)
  if func_num == 37:
    #consecutive losses sup
    a = 0.3 * mental
    b = 1.0 / 80.0
    ret = inv_down(factor,a,b)
  if func_num == 38:
    #consecutive losses ang
    a = -40.0 * ((11.0 - mental)/11.0 + 0.5)
    b = 1.0 / 80.0
    c = 0.04 * 0.8 * mental + 0.1
    ret = sig(factor,a,b,c)
  if func_num == 39:
    #consecutive losses sad
    a = -30.0 * ((11.0 - mental)/11.0 + 0.5)
    b = 1.0 / 80.0
    c = 0.1 * (3.0 - 0.2 * (11.0 - mental))
    ret = sig(factor,a,b,c)

  return ret

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  print(func_num[0][0])
  print(np.linspace(0.01,1,100))

  for i in range(40):
    val = np.zeros((10,100))
    val2 = np.zeros((10,100))
    for m in range(1,10):
      for f in range(1,100):
        #val[m][f] = func(inv_norm(i,f),m,i)
        val[m][f] = func(inv_norm(i,f/100.0),m,i)
      plt.plot(inv_norm(i,np.linspace(0.01,1,99)),val[m,1:])
    #plt.plot(val2)
    plt.title(str(i)+func_name_str[i])
    plt.show()
    """
    if i == 30 or i== 31 or i == 38:
      plt.show()
    else:
      plt.clf()
    """
