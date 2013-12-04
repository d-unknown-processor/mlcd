import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pylab import *
import math

def get_data(file_str):
  f = open(file_str, 'r').readlines()
  Ys = []
  for line in f:
    Ys.append(float(line))
  return Ys

def main():

  test1 = get_data("outputs/q6p1/1-chain.txt-testll")
  train1 = get_data("outputs/q6p1/1-chain.txt-trainll")
  test2 = get_data("outputs/q6p1/2-chain.txt-testll")
  train2 = get_data("outputs/q6p1/2-chain.txt-trainll")
  test3 = get_data("outputs/q6p1/3-chain.txt-testll")
  train3 = get_data("outputs/q6p1/3-chain.txt-trainll")

  plt.plot(xrange(len(train1)), train1, xrange(len(test1)), test1)
  plt.title('Training and Test Loglikelihood (green = test, blue = training)')
  plt.ylabel('Loglikelihood')
  plt.xlabel('Iteration')
  plt.savefig('q6_p1_1.pdf',dpi=100)
  plt.clf()

  plt.plot(xrange(len(train2)), train2, xrange(len(test2)), test2)
  plt.title('Training and Test Loglikelihood (green = test, blue = training)')
  plt.ylabel('Loglikelihood')
  plt.xlabel('Iteration')
  plt.savefig('q6_p1_2.pdf',dpi=100)

  plt.clf()
  plt.plot(xrange(len(train3)), train3, xrange(len(test3)), test3)
  plt.title('Training and Test Loglikelihood (green = test, blue = training)')
  plt.ylabel('Loglikelihood')
  plt.xlabel('Iteration')
  plt.savefig('q6_p1_3.pdf',dpi=100)
  
if __name__ == "__main__":
  main()

