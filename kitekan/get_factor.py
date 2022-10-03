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


class DataReader:
    def __init__(self,filepath,sample_rate):
        self.filepath = filepath
        self.sample_rate = sample_rate
        self.smooth_length = int(sample_rate*WINDOW_WIDTH_SEC/SMOOTH_LENGTH)
        self.window_width_length = int(sample_rate*WINDOW_WIDTH_SEC)
        self.data=np.loadtxt(self.filepath,delimiter=",")
        #===text visualizing

    def show_graph(self):
        for t in range(0,self.data.shape[0],(int)(self.window_width_length/2.0)):
          start = t
          end = start+self.window_width_length
          #print(start,end)
          sample_data = self.data[start:end,4]
          #sample_data = self.data[:,0]
          smooth_filter=np.ones(self.smooth_length)/self.smooth_length
          smooth_data = np.convolve(sample_data,smooth_filter,mode="valid")

          #if(max(smooth_data)>13):
          plt.plot(sample_data,label="raw")
          plt.plot(smooth_data,label="smooth")
          #plt.ylim(0,2)
          plt.legend()
          plt.show()

def main(username):

    data_type = "photo"
    datafile,rate_label=file_pointer.get_filename(username,data_type)

    sample_rate = int(config['SAMPLE_RATE'][rate_label])

    sensor_data=DataReader(datafile,sample_rate)
    sensor_data.show_graph()

if __name__ == '__main__':
    username="1110" 
    main(username)
