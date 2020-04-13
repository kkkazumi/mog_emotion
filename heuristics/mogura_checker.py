import numpy as np
import multiprocessing as mp

out_thre = 0.2

DATA_COL_LEN = 7
TIME_COL_LEN = 3

POP_UP = 1
SINK_DOWN = -1

#may not necessary
#MIN_COL = 8
#SEC_COL = 9
#MSEC_COL = 10


class Mog_check:
    def __init__(self, data, window_size):
        self.data = data[:,:DATA_COL_LEN]
        #self.data = np.hstack((data[:,:DATA_COL_LEN],data[:,-3:])
        self.time_data = data[:,-TIME_COL_LEN:]

        self.win_size = window_size
        self.state = np.zeros((1,DATA_COL_LEN))
        self.history = np.empty((1,DATA_COL_LEN))
        self.state_change_history= np.empty((1,2+TIME_COL_LEN))
        #self.state_change= np.zeros((1,DATA_COL_LEN))
        self.i = 0

    def get_data(self):
        self.window = self.data[self.i:self.i+self.win_size]
        self.i = self.i + 1

    def out_height(self):
        self.get_data()
        return np.mean(self.window,axis=0)

    def push_history(self):
        if(self.i==1):
          self.history = self.state
        else:
          self.history = np.append(self.history,self.state,axis=0)
        #print("history",self.i,self.history,self.state)

    def out_state(self):
        #print("True: height > threshold")
        h_data = self.out_height()
        self.state[0,:] = (h_data>out_thre).astype(np.int)
        self.push_history()
        return self.state

    def time_of_popnsink(self,ref_time,mole_num,popnsink_flg):
        if(len(self.state_change_history)>1):
          if(np.where(self.state_change_history[:,1]==mole_num)):
            if(np.where(self.state_change_history[:,0]==popnsink_flg)):
              target_i = np.max(np.where((self.state_change_history[:,1]==mole_num)&(self.state_change_history[:,0]==popnsink_flg)))
              change_time = self.state_change_history[target_i,-TIME_COL_LEN:]
              ret_time = ref_time - change_time

            else:
              ret_time = None
          else:
            ret_time = None
        else:
          ret_time = None

        print("rettime",ret_time)
        input()

        return ret_time
        

    def check_state_diff(self):
        state_change= np.zeros((1,DATA_COL_LEN))
        change_log= np.zeros((1,2+TIME_COL_LEN))

        if(self.i>3):
          now = self.history[-1]
          pre = self.history[-2]
          state_change[0,:] = now - pre

          for check_diff in [SINK_DOWN,POP_UP]:
            if(np.any(state_change==check_diff)):
              if(len(np.where(state_change==check_diff)[0])==1):
                change_log[0,:] = np.hstack((np.array([check_diff,np.where(state_change==check_diff)[1][0]]),self.time_data[self.i,-TIME_COL_LEN:]))

              self.state_change_history=np.append(self.state_change_history,change_log,axis=0)
              print("change history check",self.state_change_history.shape,self.state_change_history[-1])
        

    def out_rate(self):
        rate = float(np.sum(self.state))/7.0
        return rate

if __name__ == '__main__':
    data = np.loadtxt('test_ad_all.csv',delimiter=",")
    #data = np.loadtxt('test2.csv',delimiter=",")
    winsize = 100
    i=0
    mogura = Mog_check(data,winsize)
    test = np.array([[5000,SINK_DOWN,2,37,29,1900],[80000,POP_UP,2,37,42,1000]])
    j=0
    while(i<data.shape[0]):
        print("mogura out state",i,mogura.out_state())
        mogura.check_state_diff()
        if(i==test[j,0]):
          print(mogura.time_of_popnsink(test[j,-3:],test[j,2],test[j,1]))
          j+=1

        i=i+1
