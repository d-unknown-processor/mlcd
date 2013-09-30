import numpy as np
import matplotlib.pyplot as plt


def modelcheck(model, X, Y, w_0, noise):
    means = []
    for i in range(0, len(X) - 1):
        V_n = get_V_batch(model, X[0:i], noise)
        w_n = get_w_batch(model, V_n, noise, X[0:i], Y[0:i])
        means.append((w_n.transpose() * X[i+1].transpose())[0,0])

    return means


def get_V_batch(model, X, noise):
    A = np.add(np.multiply(noise, np.linalg.inv(model[1])), X.transpose() * X)
    return np.multiply(noise, np.linalg.inv(A))

def get_w_batch(model, V_n, noise, X, Y):
    return np.multiply((1/float(noise)), (V_n * X.transpose()) * Y)  

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
    # Initial variable for models 0, ab, and cd.
    variance_0_0 = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    variance_ab_0 = np.matrix([[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    variance_cd_0 = np.matrix([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]])

    # Initial mean or w_0
    w_0 = 0
    model_0 = (w_0, variance_0_0)
    model_ab = (w_0, variance_ab_0)
    model_cd = (w_0, variance_cd_0)

    #Evaluation model 0
    means_model_0 = modelcheck(model_0, X, Y, w_0, 4)
    
    errors_model_0 = []
    for i in range(1, len(means_model_0)):
        squared_error = ((means_model_0[0:i] - Y[0:i]) ** 2).mean()
        errors_model_0.append(squared_error)
    print "Len squared error = " + str(len(errors_model_0))
    plt.plot(range(0, 999), errors_model_0 , 'ro')
    plt.show()
    
if __name__ == "__main__":
    main()
