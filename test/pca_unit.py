import numpy as np

def comp_dim(data):
  from sklearn.decomposition import PCA
  pca = PCA()
  pca.fit(data)
  feature = pca.transform(data)

  return feature

if __name__ == "__main__":
  from mpl_toolkits.mplot3d import Axes3D
  from matplotlib import pyplot as plt
  fig = plt.figure()
  ax = Axes3D(fig)

  load_data = np.loadtxt('test_imu.csv',delimiter=",")
  data = load_data[:,:6]

  label = np.ones(data.shape[0])
  label[:5800] = label[:5800] * 0
  label[6041:6800] = label[6041:6800] * 0
  label[7100:] = label[7100:] * 0 

  feature = comp_dim(data)

  plt.scatter(feature[:,0],feature[:,1],feature[:,2],c=list(label))
  #for 2D
  #plt.scatter(feature[:,0],feature[:,1],c=list(label))
  plt.grid()
  plt.show()
