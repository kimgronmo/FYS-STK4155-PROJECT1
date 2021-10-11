import sklearn.linear_model as skl


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from support_functions import *
from printer_functions import *
from analysis_functions import *

# imports functions for real world data
from imageio import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image

def exercise1():
    print("\n")
    print("#"*80)
    # Setting up data files and directories
    #initiate_data_structure()
    # Setting up initial data sets

    print("Starting exercise1:")
    # data has already been scaled..
    betaValues = OLS(X_train,y_train)
    #print(X_train.shape,X_test.shape,y_train.shape,y_test.shape,z.shape)
    #print("betavalues for real world data size is :",betaValues.size)    
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    
    print("Checking if my code works correctly:")
    print("My MSE is: ",MSE(y_test,ypredict))
    print("My R2 score is: ",R2(y_test,ypredict))
    
    print("\nTesting against sklearn OLS: ")
    clf = skl.LinearRegression(fit_intercept=False).fit(X_train, y_train)
    print("MSE after scaling: {:.4f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score after scaling {:.4f}".format(clf.score(X_test,y_test)))
    
    # Calculates and prints the beta confidence intervals
    betaConfidenceIntervals(X_train,betaValues,y_train,ytilde)
    print("\n""\n")

def exercise2(x,y,z,maxPolyDegree,polynomial,n_bootstraps,error,bias,variance,nlambdas,lambdas):
    print("\n")
    print("#"*80)
    print("Starting exercise2:")

    # Generating figure 2.11 from Hastie..
    print("\nGenerating figure 2.11 from Hastie..")
    figure211(x,y,z)
    
    print("\nComputing bias and variance trade-off as a function of model complexity\n")
    biasVariance("OLS",maxPolyDegree,polynomial,x,y,z,n_bootstraps,error,bias,variance, \
        nlambdas,lambdas)

def exercise3(x,y,z):
    print("\n")
    print("#"*80)
    print("Starting exercise3:")
    print("\nStarting Cross-validation\n")
    crossValidation(5,"OLS",x,y,z)    

def exercise4(maxPolyDegree,polynomial,x,y,z,n_bootstraps,error,bias,variance,nlambdas,lambdas):
    print("\n")
    print("#"*80)
    print("Starting exercise4:")
    print("\nStarting Ridge regression\n")
    biasVariance("Ridge",maxPolyDegree,polynomial,x,y,z,n_bootstraps,error,bias,variance, \
        nlambdas,lambdas)
    crossValidation(5,"Ridge",x,y,z)    

def exercise5(X_train,X_test,lambdas,y_train,y_test,nlambdas):
    print("\n")
    print("#"*80)
    print("Starting exercise5:")
    print("\nStarting Lasso regression\n")
    LassoRegression(X_train,X_test,lambdas,y_train,y_test,"franke",nlambdas)

# Needs an image file in "./DataFiles"
def exercise6():
    print("\n")
    print("#"*80)
    print("Starting exercise6:")
    print("\nStarting Analysis of Real World Data\n")
    DATA_ID = "DataFiles/"
    def data_path(dat_id):
        return os.path.join(DATA_ID, dat_id)
    
    terrain1 = Image.open(data_path("SRTM_data_Norway_1.tif"), mode='r')
    terrain1.mode = 'I'    

    # Problem making the design matrix when the image is not a square
    x = np.linspace(0, 1, int(terrain1.size[0]/5))
    y = np.linspace(0, 1, int(terrain1.size[0]/5))
    z_temp = np.array(terrain1)
    z_temp = z_temp - np.min(z_temp)
    z_temp = z_temp / np.max(z_temp)
    z = np.empty([int(terrain1.size[0]/5),int(terrain1.size[0]/5)])
    for i in range(int(terrain1.size[0]/5)):
        for j in range(int(terrain1.size[0]/5)):
            z[i,j]=z_temp[i,j]

    z = z.flatten()
    z=z[:y.size]
    X = create_X(x, y, n=5)

    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)
    
    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    error = np.zeros(maxPolyDegree)
    bias = np.zeros(maxPolyDegree)
    variance = np.zeros(maxPolyDegree)
    polynomial = np.zeros(maxPolyDegree) 


    ## Exercise 1: ##
    betaValues = OLS(X_train,y_train)

    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    
    print("Checking if my code works correctly:")
    print("My MSE is: ",MSE(y_test,ypredict))
    print("My R2 score is: ",R2(y_test,ypredict))
    print("\nTesting against sklearn OLS: ")
    clf = skl.LinearRegression(fit_intercept=False).fit(X_train, y_train)
    print("MSE after scaling: {:.4f}".format(mean_squared_error(clf.predict(X_test), y_test)))
    print("R2 score after scaling {:.4f}".format(clf.score(X_test,y_test)))
    
    # Calculates and prints the beta confidence intervals
    betaConfidenceIntervals(X_train,betaValues,y_train,ytilde)
    print("\n""\n")
    
    ## Exercise 2: ##
    figure211(x,y,z)
    print("\nComputing bias and variance trade-off as a function of model complexity\n")
    biasVariance("OLS",maxPolyDegree,polynomial,x,y,z,n_bootstraps,error,bias,variance, \
        nlambdas,lambdas)

    ## Exercise 3: ##
    crossValidation(5,"OLS",x,y,z)
    
    ## Exercise4: ##
    biasVariance("Ridge",maxPolyDegree,polynomial,x,y,z,n_bootstraps,error,bias,variance, \
        nlambdas,lambdas)
    crossValidation(5,"Ridge",x,y,z)

    ## Exercise 5: ##
    print("\nLASSO regression")
    LassoRegression(X_train,X_test,lambdas,y_train,y_test,"realworld",nlambdas)    

    #plotting terrain data
    #plt.figure()
    #plt.title("Terrain data Norway")
    #plt.imshow(terrain1, cmap="gray")
    #plt.show()
 
    # Summary of regression results: Real World Data
    summaryRegression(X_train,X_test,lambdas,y_train,y_test,nlambdas)     

if __name__ == '__main__':
    # Starting project 1

    # Setting up data files and directories
    initiate_data_structure()
    X_train,X_test,y_train,y_test,lambdas,maxPolyDegree,x,y,z, \
        error,bias,variance,polynomial,n_bootstraps,nlambdas = create_dataset()
    
    exercise1()
    exercise2(x,y,z,maxPolyDegree,polynomial,n_bootstraps,error,bias,variance,nlambdas,lambdas)
    exercise3(x,y,z)
    exercise4(maxPolyDegree,polynomial,x,y,z,n_bootstraps,error,bias,variance,nlambdas,lambdas)
    exercise5(X_train,X_test,lambdas,y_train,y_test,nlambdas)

    # Summary of regression results: Franke Function
    summaryRegression(X_train,X_test,lambdas,y_train,y_test,nlambdas)

    # Note if exercises 1-5 is run before exercise 6 the output images
    # in exercise 6 might overwrite the figures (both show franke results and 
    # real world data results on same figure)
    exercise6()