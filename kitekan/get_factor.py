import numpy as np
import configparser

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

    def readdata(self):
        self.data=np.loadtxt(self.filepath,delimiter=",")

        #===text visualizing
        print(self.data.shape[0])
        for t in range(0,self.data.shape[0],(int)(self.window_width_length/2.0)):
          start = t
          end = start+self.window_width_length
          #print(start,end)
          sample_data = self.data[start:end,2]
          #sample_data = self.data[:,0]
          smooth_filter=np.ones(self.smooth_length)/self.smooth_length
          smooth_data = np.convolve(sample_data,smooth_filter,mode="valid")

          if(max(smooth_data)>13):
            plt.plot(sample_data,label="raw")
            plt.plot(smooth_data,label="smooth")
            plt.ylim(-15,15)
            plt.legend()
            plt.show()

def main():
    imu_filedata = "../1110/new_exp/21091010-175046/imu_data_21091010-175110.csv"
    #toriaezu toriaez.

    imu_sample_rate = int(config['SAMPLE_RATE']['imu_rate'])

    imu=DataReader(imu_filedata,imu_sample_rate)
    imu.readdata()

if __name__ == '__main__':
    main()
