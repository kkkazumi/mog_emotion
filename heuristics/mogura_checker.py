import numpy as np

out_thre = 2.0


class Mog_check:
    def __init__(self, data, window_size):
        self.data = data
        #self.window =
        self.win_size = window_size
        self.state = np.zeros((1,7))
        self.history = np.empty((1,7))
        self.i = 0

    def get_data(self):
        self.window = data[self.i:self.i+self.win_size]
        self.i = self.i + 1

    def out_height(self):
        self.get_data()
        return np.mean(self.window,axis=0)

    def push_history(self):
        self.history = np.append(self.history,self.state,axis=0)
        print(self.history,self.state)

    def out_state(self):
        print("True: height > threshold")
        h_data = self.out_height()
        self.state[0,:] = (h_data>out_thre).astype(np.int)
        return self.state

    def out_rate(self):
        rate = float(np.sum(self.state))/7.0
        return rate

if __name__ == '__main__':
    data = np.loadtxt('test2.csv',delimiter=",")
    winsize = 4
    i=0
    mogura = Mog_check(data,winsize)
    while(i<data.shape[0]):
        print(i,mogura.out_state())
        print(mogura.push_history())
        raw_input()

        i=i+1
