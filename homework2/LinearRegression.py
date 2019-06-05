#-*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt


def linearRegression(alpha=0.01,num_iters=400):
    print(u"loading the data...\n")
    
    data = loadtxtAndcsv_data("data.txt",",",np.float64)  #load the data
    X = data[:,0:-1]
    y = data[:,-1]
    m = len(y)
    col = data.shape[1]
    
    X,mu,sigma = featureNormaliza(X)
    plot_X1_X2(X)
    
    X = np.hstack((np.ones((m,1)),X))
    
    print(u"\nExecute the gradiant descent....\n")
    
    theta = np.zeros((col,1))
    y = y.reshape(-1,1)
    theta,J_history = gradientDescent(X, y, theta, alpha, num_iters)
    
    plotJ(J_history, num_iters)
    
    return mu,sigma,theta   # get the mu, sigma and theta
    
   
# load txt and csv file
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)


# Do feature Normalization
def featureNormaliza(X):
    X_norm = np.array(X)
    # define the key
    mu = np.zeros((1,X.shape[1]))   
    sigma = np.zeros((1,X.shape[1]))
    
    mu = np.mean(X_norm,0)          # get the mean
    sigma = np.std(X_norm,0)        # get the std
    for i in range(X.shape[1]):     # go through the columns
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]  # normalization
    
    return X_norm,mu,sigma

def plot_X1_X2(X):
    plt.scatter(X[:,0],X[:,1])
    plt.show()


# Do the gradiant descent algorithm
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)      
    n = len(theta)
    
    temp = np.zeros((n,num_iters))   # store the theta，convert this to th matrix
    
    
    J_history = np.zeros((num_iters,1)) # record the cost function for each iteration
    
    for i in range(num_iters):

        ############################################################################
        # TODO: Calculate the gradiant descent    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    return theta,J_history  

# compute the cost function
def computerCost(X,y,theta):
    m = len(y)
    J = 0

    ############################################################################
    # TODO: Calcuate the cost function    #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return J

# plot the iteration for each changing
def plotJ(J_history,num_iters):
    x = np.arange(1,num_iters+1)
    plt.plot(x,J_history)
    plt.xlabel(u"iteration time")
    plt.ylabel(u"cost value")
    plt.title(u"The variation of the cost function")
    plt.show()

# test linearRegression
def testLinearRegression():
    mu,sigma,theta = linearRegression(0.01,400)
    
# predict the result
def predict(mu,sigma,theta):
    result = 0
    # normalization
    predict = np.array([1650,3])
    norm_predict = (predict-mu)/sigma
    ############################################################################
    # TODO: Computer the result here    #
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return result
    
if __name__ == "__main__":
    testLinearRegression()