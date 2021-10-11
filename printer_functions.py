import numpy as np
import matplotlib.pyplot as plt

from analysis_functions import *
from support_functions import *

def figure211(x,y,z):
    # makes figure 2.11 from Hastie..
    print("...this might take a wile... change maxPoly to speed up..")
    maxPoly = 50
    numPoly = np.zeros(maxPoly)
    mse_train211=[]
    mse_test211=[]
    for i in range(maxPoly):
        # Trying to print a progress bar...
        #print("#",end=''))
        numPoly[i] = i
        X = create_X(x, y, n=i)

        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)        

        betaValues = OLS(X_train,y_train)
        ytilde = X_train @ betaValues
        ypredict = X_test @ betaValues 
        mse_train211.append(MSE(y_train,ytilde))
        mse_test211.append(MSE(y_test,ypredict))

    # Generates plot and saves it in folder
    plt.figure(1)
    plt.title("Test and training error as a function of model complexity", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"mean square error", fontsize=10)
    plt.plot(numPoly, mse_train211, label = "MSE training")
    plt.plot(numPoly, mse_test211, label = "MSE test data")
    plt.legend([r"mse from training data", r"mse from test data"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'figure211 from Hastie sample 1000.png'), transparent=True, bbox_inches='tight')
    #plt.show()   