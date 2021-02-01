from sklearn import datasets
from widrow_hoff_learning_algorithm import widrow_hoff_algorithm

# TRAINING DATA:
# Some of the exercises the Iris dataset. This dataset contains 150 samples from 3 classes. 
# Each sample is a four dimensional feature vector. This data can be obtained as follows.

# The structure "iris" contains a field iris.data that is an array, each row of which
# is a feature vector for one sample, and a field iris.target that is a vector defining
# the class labels associated with the corresponding rows of iris.data.
iris = datasets.load_iris()

# INITIALISING VARIABLES AND CALLING WIDROW-HOFF LEARNING ALGORITHM FUNCTION
a = [1.0,0.0,0.0]
b = [1.0,0.5,0.5,1.5,0.5,1.0]
n = 0.1
epoch = 2
X = [[1],[1],[1],[-1],[-1],[-1]]
Y = [[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]]

widrow_hoff_algorithm(a,b,n,epoch,Y,X)
