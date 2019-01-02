import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("iris.csv", delimiter=",")
for i in range(0,4):
    X = data[:,i]
    print(X)
    plt.hist(X)
    plt.show()
Y = data[:,4]
print(Y)
plt.hist(Y)
plt.show()
