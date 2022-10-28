import numpy as np
import configparser

import file_pointer

config_file = "config.ini"
config = configparser.ConfigParser()
config.read(config_file)

WINDOW_WIDTH_SEC = float(config['WINDOW_SIZE']['factor_estimation_period_sec'])#might be 0.5
SMOOTH_LENGTH = float(config['WINDOW_SIZE']['smoothing_length'])#might be 100

#debugggg
import matplotlib.pyplot as plt
#debuggg coco made

def dummy_factors(length=100,dim=5):
    factors=np.random.randint(0,10,(length,dim))/10.0
    return factors

def dummy_signals(length=100,dim=4)
    signals=np.random.randint(0,10,(length,dim))/10.0
    return signals

class DataReader:
    def __init__(self,filepath,sample_rate):
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.smooth_length = int(sample_rate*WINDOW_WIDTH_SEC/SMOOTH_LENGTH)
        self.window_width_length = int(sample_rate*WINDOW_WIDTH_SEC)
        self.data=np.loadtxt(self.filepath,delimiter=",")
        #===text visualizing

    def draw_graph(self,sample_data):
        
      smooth_filter=np.ones(self.smooth_length)/self.smooth_length
      smooth_data = np.convolve(sample_data,smooth_filter,mode="valid")
      plt.plot(sample_data,label="raw")
      plt.plot(smooth_data,label="smooth")
      #plt.ylim(0,2)
      plt.legend()
      plt.show()

    def show_graph(self,mode):
        if(mode=="ALL"):
          sample_data = self.data[:,4]
          self.draw_graph(sample_data)
        else:
          for t in range(0,self.data.shape[0],(int)(self.window_width_length/2.0)):
            start = t
            end = start+self.window_width_length
            sample_data = self.data[start:end,4]
            self.draw_graph(sample_data)

    def show_data_histogram(self):

        for i in range(7):
            sample_data = self.data[:,i]
            smooth_filter=np.ones(self.smooth_length)/self.smooth_length
            smooth_data = np.convolve(sample_data,smooth_filter,mode="valid")
            print(np.unique(smooth_data))

            hist,b=np.histogram(smooth_data,bins=100)
            bins = []
            for i in range(1, len(b)):
                  bins.append((b[i-1]+b[i])/2)
            plt.bar(bins,hist)
            plt.show()

def main(username):

    data_type = "imu"
    #data_type = "photo"
    datafile,rate_label=file_pointer.get_filename(username,data_type)

    sample_rate = int(config['SAMPLE_RATE'][rate_label])
    print("input mode to show graph from ALL or not")
    mode=input()

    sensor_data=DataReader(datafile,sample_rate)
    sensor_data.show_graph(mode)
    #sensor_data.show_data_histogram()

if __name__ == '__main__':
    username="1110" 
    main(username)
