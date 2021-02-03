import numpy as np
from prettytable import PrettyTable
from sklearn import preprocessing

# SEQUENTIAL WIDROW HOFF LEARNING ALGORITHM BASED MINIMISATION
# - Set values of the hyper-parameters(η and b)
# - Initialise a to arbritrary solution
# - For each sample, yk in the dataset in turn
#   -> update solution: a←a+η(bk−at yk) yk

def two_class_widrow_hoff_algorithm(a, b, n, epoch, Y, X, class_1 = [[]], class_neg_1 = [[]], testing = False):
    result = [] # Defining a list of results, used later for displaying in a table
    store_a = [] # Storing newely generated a to use in calculatation of g > 0

    if not isinstance(Y, list) and not isinstance(X, list): Y,X = Y.tolist(), X.tolist() # Converting datasets to list if not already
    if any(isinstance(el, list) for el in X) == False : X = [[x] for x in X] # Converting X data into a list of lists if not already
    for i in range(len(Y)): Y[i].insert(0, 1.0) # Adding "1.0" into the first position of every element of Y used for normalisation

    # BASED ON CLASS CATERGORISATION INPUTTED, THE X VALUES ARE CONVERTED TO CLASS 1 OR -1
    if len(class_1) and len(class_neg_1) != 0:
        for iter0 in range(len(X)):
            for x in range(len(X[iter0])):
                if X[iter0][x] in class_neg_1:
                    X[iter0][x] = -1
                elif X[iter0][x] in class_1: 
                    X[iter0][x] = 1

    for iter1 in range(epoch): # Iterating over epoch
        for iter2 in range(len(Y)): # Iterating through ever element within Y
            # STEP 1: NORMALISING DATA
            y = np.multiply(Y[iter2], X[iter2]) if X[iter2] == [-1] else Y[iter2]

            # STEP 2: CALCULATING: "aTyk"
            ay = np.dot(a, y)

            # STEP 3: CALCULATING UPDATE: "aTnew=aT+η(bk−aTyk)yTk"
            update = np.zeros(len(y))
            for iter3 in range(len(y)):
                update[iter3] = n * (b[iter2] - ay) * y[iter3] # η(bk−at yk) yk

            # STEP 4: ADDING UPDATED PARTS TO A
            a = np.add(a, update) # a←a+η(bk−at yk) yk

            # APPENDING RESULTS TO ARRAY, SPECIFICALLY FOCUSING ON 3 COLUMNS, ITERATIONS, AY AND A_NEW
            result.append((str(iter2 + 1 + (len(Y) * iter1)),np.round(ay, 2),np.round(a, 2)))
            #print("THIS IS THE VALUE OF a: " + str(a))
            store_a.append(a)

    # THIS CODE IS USED TO CALCULATE VALUES OF A[[]] WHICH ARE > 0
    counter = 0 # Initialising counter variable
    negcounter = 0
    for index0 in range(len(store_a)): # Iterating over the final list of newely generated a
        for index1 in range(len(store_a[index0])): # Iterating over ever element for every index of a
            if store_a[index0][index1] > 0: # Checking if value is > 0
                counter = counter + 1 # Iterating counter if value is > 0
            else: 
                negcounter = negcounter + 1
    percentage = (counter/(len(store_a)*len(store_a[0])))*100 # Calculating percentage of values > 0
    print(str(np.round(percentage, 2))+"%") # Printing percentage
    
    # Defining a table to display results
    results_table = PrettyTable(('iteration', 'ay', 'a_new'))
    
    for row in result: results_table.add_row(row)
    results_table.align['Iteration'] = 'c'
    results_table.align['ay'] = 'l'
    results_table.align['a_new'] = 'l'

    print(results_table)

    store_a = [i.tolist() for i in store_a]
    return store_a[-1]


