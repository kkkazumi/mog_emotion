import scipy.optimize as opt
import numpy as np

##time
f = lambda x : 5.0*x[0]**2.0 - 6.0*x[0]*x[1] + 5.0*x[1]**2 - 10.0*x[0] + 6.0*x[1] + 3.0*x[2]
g = lambda x : np.array( [ 10.0 * x[0] - 6.0 *x[1] -10.0, 10.0 * x[1] - 6.0 *x[0] + 6.0 ] )

out2 = []
def callback(xk):
  out2.append(xk)
  print(xk)
          
# initial value
x = np.array( [-2.5, 0.0,3.0] )
out2.append(x)
xx, y, d = opt.fmin_l_bfgs_b(f, x, fprime=g, m=5, iprint=0, epsilon=10.0**(-8), maxiter=20, maxls=1, callback=callback)
