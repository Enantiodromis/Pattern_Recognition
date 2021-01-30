import numpy as np
from prettytable import PrettyTable

# DEFINING FUNCTION FOR WIDROW-HOFF LEARNING ALGORITHM:
# The Widrow-Hoff Learning Algorithm is applied to the following data in order to learn the parameters 
# of a linear discriminant function:
# 
# Feature vector, xT: [[0, 2], [1, 2], [2, 1], [-3, 1], [-2, 1], [-3, 2]]
# Class: [1, 1, 1, -1, -1, -1]
#
# 2 epochs of the Sequential Widrow-Hoff Learning Algorithm were applied to the training data
# The initial parameters values of aT = (w0, wT) = (1.0,0.0,0.0), margin vector bT = (1.0,0.5,0.5,1.5,0.5,1.0) 
# and a learning rate of 0.1 were used.

def widrow_hoff_algorithm(a, b, n, epoch, Y):
    result = []
    for iter1 in range(epoch):
        for iter2 in range(len(Y)):
            a_prev = a
            y = Y[iter2]

            # CALCULATING: "aTyk"
            ay = np.dot(a, y)

            # CALCULATING UPDATE: "aTnew=aT+η(bk−aTyk)yTk"
            update = np.zeros(len(y))
            for iter3 in range(len(y)): 
                update[iter3] = n * (b[iter2] - ay) * y[iter3]
            # ADDING UPDATED PARTS TO A
            a = np.add(a, update)

            # APPENDING RESULTS TO ARRAY, SPECIFICALLY FOCUSING ON 3 COLUMNS, ITERATIONS, AY AND A_NEW
            result.append((str(iter2 + 1 + (len(Y) * iter1)),np.round(ay, 4), np.round(a, 4)))
            
    results_table = PrettyTable(('iteration', 'ay', 'a_new'))
    for row in result: results_table.add_row(row)

    results_table.align['Iteration'] = 'c'
    results_table.align['ay'] = 'l'
    results_table.align['a_new'] = 'l'

    print(results_table)