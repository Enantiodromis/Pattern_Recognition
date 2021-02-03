import numpy as np
from sklearn import datasets
from widrow_hoff_learning_algorithm import two_class_widrow_hoff_algorithm

# QUESTION 1: Apply 2 epochs of the Sequential Widrow-Hoff Learning Algorithm to the training data.
#
# Use initial parameters values:
# aT=(w0,wT)=(1.0,0.0,0.0)
# Margin vector bT=(1.0,0.5,0.5,1.5,0.5,1.0), 
# A learning rate of 0.1. 

#Initialising inputs and calling the widrow_hoff_learning_algotithm function.
a_1 = [1.0,0.0,0.0]
b_1 = [1.0,0.5,0.5,1.5,0.5,1.0]
n_1 = 0.1
epoch_1 = 2
X_1 = [[1],[1],[1],[-1],[-1],[-1]]
Y_1 = [[0.0, 2.0], [1.0, 2.0], [2.0, 1.0], [-3.0, 1.0], [-2.0, -1.0], [-3.0, -2.0]]
two_class_widrow_hoff_algorithm(a_1,b_1,n_1,epoch_1,Y_1,X_1)

# TRAINING DATA:
# The Iris dataset. This dataset contains 150 samples from 3 classes. 
# Each sample is a four dimensional feature vector.
# The structure "iris" contains a field iris.data that is an array, each row of which
# is a feature vector for one sample, and a field iris.target that is a vector defining
# the class labels associated with the corresponding rows of iris.data.

# Loading the iris dataset, a import also included
iris = datasets.load_iris()

# QUESTION 2: Use the Widrow-Hoff Learning Algorithm to find the parameters of a 
# linear discriminant function that should output g(x)>0 for class 0 and g(x)≤0 for the other two classes of the iris dataset.
# Apply 2 epochs of the Sequential Widrow-Hoff Learning Algorithm. 
#
# Use initial parameter values:
# -> aT=(w0,wT)=(0.5,0.5,−2.5,1.5,−0.5)
# -> Margin vector b in which all values are equal to 1 the length b == length of element of Y
# -> A learning rate of 0.01.
#
# Calculate the percentage of samples for which the linear discriminant function produces the desired output 
# (i.e. g(x)>0 for the 1st 50 samples, and g(x)≤0 for the remaining 100 samples) using the initial parameter values, and those calculated after learning for 2 epochs.

# Initialising inputs and calling the widrow_hoff_learning_algotithm function.
a_2 = [0.5,0.5,-2.5,1.5,-0.5]
b_2 = [1.0 for i in range(150)]
n_2 = 0.01
epoch_2 = 2
X_2 = iris.target
Y_2 = iris.data
C1_2 = [0]
CN1_2 = [1,2]

training_algo = two_class_widrow_hoff_algorithm(a_2,b_2,n_2,epoch_2,Y_2,X_2,C1_2,CN1_2)
two_class_widrow_hoff_algorithm(training_algo,b_2,n_2,epoch_2,Y_2,X_2,C1_2,CN1_2, testing = True)
