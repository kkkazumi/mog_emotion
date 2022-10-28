import numpy as np



def argmax_ndim(arg_array):
  return np.unravel_index(arg_array.argmax(), arg_array.shape)

def argmin_ndim(arg_array):
  return np.unravel_index(arg_array.argmin(), arg_array.shape)

def find_grad(step,funcs,args):
  func_g, func_h = funcs
  factor, mental, signal = args

  if(len(mental)>1):
    time_length = len(mental[0])
    dim=len(mental)
  else:
    time_length=len(mental)
    mental = np.reshape(mental,(-1,time_length))
    dim=1
  #print(time_length)
  err_array = np.zeros((dim,time_length))

  print(mental)

  for t in range(time_length):
    for i in range(dim):
      mental[i,t] += step
      err_array[i,t] = dim*t

  return argmin_ndim(err_array)


if __name__ == '__main__':
  time_len = 10
  mental_dim = 2
  step = 0.1
  args = 0

  f_dummy=np.random.randint(0,10,time_len)
  s_dummy=np.random.randint(0,10,time_len)

  mental=(np.random.randint(0,10,(mental_dim,time_len))-5*np.ones((mental_dim,time_len)))/10.0

  args = f_dummy,mental,s_dummy

  funcs = argmax_ndim,argmin_ndim

  print(find_grad(step,funcs,args))
