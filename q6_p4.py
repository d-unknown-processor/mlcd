import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pylab import *
import math

def main():

  #Ys = [math.exp(-2.2438569852346e+05), math.exp(-2.2543211834995e+05), math.exp(-2.2638102346694e+05), math.exp(-2.2779942648635e+05), math.exp(-2.3046844322923e+05)]
  Ys = [-2.2438569852346, -2.2543211834995,-2.2638102346694,-2.2779942648635,-2.3046844322923]
  print Ys
  Xs = [10,20,30,40,50]
  Xs.reverse()
  num_bins = len(Ys)
  # the histogram of the data
  plt.figure(1)
  ax = plt.subplot(211)
  ax.bar(arange(len(Ys)),Ys, width=0.3)
  plt.xticks(xrange(len(Xs)), Xs, ha='center')
  plt.xlabel('Topic Sizes')
  plt.ylabel('Loglikelihood/e+05')
  ax.set_ylim([min(Ys), max(Ys) +.1])
  #ax.set_xlim([1,5])
  plt.title('Loglikelihood of Test Set Topic Sizes')
  
  # Tweak spacing to prevent clipping of ylabel
  #plt.subplots_adjust(left=0.15)
  plt.savefig('q6_p4.pdf',dpi=100)
  
if __name__ == "__main__":
  main()

