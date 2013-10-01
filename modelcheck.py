# modelcheck.py 
# Authors: Nely Jimenez and David Snyder

import numpy as np
import matplotlib.pyplot as plt


# Returns the estimated means and sigmas at each x and y in X and Y
# for a given model, initial mean w_0, and the noise in the data.
def modelcheck(model, X, Y, w_0, noise):
    means = []
    sigmas = []
    for i in range(0, len(X) - 1):
        V_n = get_V_batch(model, X[0:i], noise)
        w_n = get_w_batch(model, V_n, noise, X[0:i], Y[0:i])
        sigmas.append(get_sigma_x(noise, V_n, X[i+1])[0,0])
        means.append((w_n.transpose() * X[i+1].transpose())[0,0])
    return means, sigmas


# Returns the covariance matrix V_n given the model, the data X and the noise.
def get_V_batch(model, X, noise):
    A = np.add(np.multiply(noise, np.linalg.inv(model[1])), X.transpose() * X)
    return np.multiply(noise, np.linalg.inv(A))

# Returns the weight vector given the model, the covariance matrix V_n,
# the noise, and the data X and Y.
def get_w_batch(model, V_n, noise, X, Y):
    return np.multiply((1/float(noise)), (V_n * X.transpose()) * Y)

# Returns the value of sigma(x) given the noise in the data, a data point
# x in X and the current covarianced matrix V_n.
def get_sigma_x(noise, V_n, x):
    return noise + x * V_n * x.transpose()

# Parses data from the file hw1a_data.csv. Returns a matrices X and Y.
def get_data():
    file_name = "hw1a_data.csv"
    lines = open(file_name, 'r').read().split("\n")
    X = []
    Y = []
    for i in range(1, len(lines)):
        line = lines[i].strip().split(",")
        if len(line) < 6:
            continue
        x = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
        y = float(line[5])
        X.append(x)
        Y.append(y)
    X = np.matrix(X)
    Y = np.matrix(Y).transpose()
    return X, Y

def main():
    X, Y = get_data()
    # Initial covariance for models 0, AB, and CD.
    # Note that variance_ab_0 encodes the correlation between stocks A and B and 
    # variance_cd_0 encodes the correlation between stocks C and D.
    variance_0_0 = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    variance_ab_0 = np.matrix([[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    variance_cd_0 = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]])

    # Initial mean or w_0
    w_0 = 0
    model_0 = (w_0, variance_0_0)
    model_ab = (w_0, variance_ab_0)
    model_cd = (w_0, variance_cd_0)

    # Find the projected means and sigmas for model 0, model AB and model CD.
    means_model_0, sigmas_model_0 = modelcheck(model_0, X, Y, w_0, 4)
    means_model_ab, sigmas_model_ab = modelcheck(model_ab, X, Y, w_0, 4)
    means_model_cd, sigmas_model_cd  = modelcheck(model_cd, X, Y, w_0, 4)
    
    # Prints the following columns:
    # mean(model 0), variance(model 0), mean(model AB), variance(model AB), mean(model CD), variance(model CD), actual y_i
    # In particular, compare the mean values predicted by each model with the actual y values.
    s = ""
    for i in range(0, len(means_model_0)):
        s += str(means_model_0[i]) + "," + str(sigmas_model_0[i]) + "," + str(means_model_ab[i]) + "," + str(sigmas_model_ab[i]) + "," + str(means_model_cd[i]) + "," + str(sigmas_model_cd[i]) + "," + str(Y[i,0]) + "\n"
    print s
    
if __name__ == "__main__":
    main()
