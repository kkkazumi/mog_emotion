import numpy as np

out_thre = 2.0


class Mog_check:
    def __init__(self, data):
        self.window = data

    def out_height(self):
        return np.mean(self.window,axis=0)

    def out_state(self):
        print("True: height > threshold")
        h_data = self.out_height()
        state = h_data>out_thre
        return state
        

if __name__ == '__main__':
    data = np.loadtxt('test2.csv',delimiter=",")
    winsize = 4
    i=0
    while(i<data.shape[0]):
        mogura = Mog_check(data[i:i+winsize])
        print(mogura.out_state())
        print(i)

        i=i+1
