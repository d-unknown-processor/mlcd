import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pylab import *
import math

def main():

  Ys = [-2.2755317454301,-2.2702829277956,-2.2682897653261,-2.2637548801664,-2.2539912306826]
  print Ys
  Xs = [0,.25,.5, .75, 1.0]
  Xs.reverse()
  num_bins = len(Ys)
  # the histogram of the data
  plt.figure(1)
  ax = plt.subplot(211)
  ax.bar(arange(len(Ys)),Ys, width=0.3)
  plt.xticks(xrange(len(Xs)), Xs, ha='center')
  plt.xlabel('Lambda')
  plt.ylabel('Loglikelihood/e+05')
  ax.set_ylim([min(Ys), max(Ys) +.1])
  #ax.set_xlim([1,5])
  plt.title('Loglikelihood of Test Set Over Lambda')
  
  # Tweak spacing to prevent clipping of ylabel
  #plt.subplots_adjust(left=0.15)
  plt.savefig('q6_p5.pdf',dpi=100)
  
if __name__ == "__main__":
  main()

