#-*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

def linearRegression():
    print(u"load...\n")
    data = loadtxtAndcsv_data("data.txt",",",np.float64)  #read the data
    X = np.array(data[:,0:-1],dtype=np.float64)
    y = np.array(data[:,-1],dtype=np.float64)
        
    # Normalization
    scaler = StandardScaler()   
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(np.array([1650,3]))
    
    # Train the linear regression model  linear_model.LinearRegression()

    ############################################################################
    # TODO: Calculate the gradiant descent    #
    # You should use sklearn to do this using linear_model.LinearRegression()
    # Take a look at this link which can tell you how to use it:
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    ############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    model = linear_model.LinearRegression()
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    #Provide the result
    result = model.predict(x_test)
    print(model.coef_)       # Coefficient of the features
    print(model.intercept_)  # bias,
    print(result)            # predict the result


# load txt
def loadtxtAndcsv_data(fileName,split,dataType):
    return np.loadtxt(fileName,delimiter=split,dtype=dataType)



if __name__ == "__main__":
    linearRegression()