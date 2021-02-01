import numpy as np
from prettytable import PrettyTable
from sklearn import preprocessing

# SEQUENTIAL WIDROW HOFF LEARNING ALGORITHM BASED MINIMISATION
# - Set values of the hyper-parameters(η and b)
# - Initialise a to arbritrary solution
# - For each sample, yk in the dataset in turn
#   -> update solution: a←a+η(bk−at yk) yk

def widrow_hoff_algorithm(a, b, n, epoch, Y, X):
    result = [] # Defining a list of results, used later for displaying in a table.

    if (isinstance(Y, list) == False and isinstance(X, list) == False): Y,X = Y.tolist(), X.tolist()
    if any(isinstance(el, list) for el in X) == False : X = [[x] for x in X]
    for i in range(len(Y)): Y[i].insert(0, 1.0) # Adding "1.0" into the first position of every element of Y used for normalisation

    for iter1 in range(epoch): # Iterating over epoch
        for iter2 in range(len(Y)): # Iterating through ever element within Y
            # STEP 1: NORMALISING DATA
            y = np.multiply(Y[iter2], X[iter2])
         
            # STEP 2: CALCULATING: "aTyk"
            ay = np.dot(a, y)
    
            # STEP 3: CALCULATING UPDATE: "aTnew=aT+η(bk−aTyk)yTk"
            update = np.zeros(len(y))
            for iter3 in range(len(y)):
                update[iter3] = n * (b[iter2] - ay) * y[iter3] # η(bk−at yk) yk
            
            # STEP 4: ADDING UPDATED PARTS TO A
            a = np.add(a, update) # a←a+η(bk−at yk) yk

            # APPENDING RESULTS TO ARRAY, SPECIFICALLY FOCUSING ON 3 COLUMNS, ITERATIONS, AY AND A_NEW
            result.append((str(iter2 + 1 + (len(Y) * iter1)),np.round(ay, 2), np.round(a, 2)))
    
    # Defining a table to display results
    results_table = PrettyTable(('iteration', 'ay', 'a_new'))
    
    for row in result: results_table.add_row(row)
    results_table.align['Iteration'] = 'c'
    results_table.align['ay'] = 'l'
    results_table.align['a_new'] = 'l'

    print(results_table)