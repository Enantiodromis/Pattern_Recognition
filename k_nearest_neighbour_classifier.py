from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def k_nearest_neighbour(irisFeatures, irisClasses, samples):
    #PUT YOUR OWN SAMPLE FEATURE VECTORS HERE
    neighbours1 = KNeighborsClassifier(n_neighbors=1)
    neighbours5 = KNeighborsClassifier(n_neighbors=5)

    neighbours1.fit(irisFeatures, irisClasses)
    print(neighbours1.predict(samples))
    neighbours5.fit(irisFeatures, irisClasses)
    print(neighbours5.predict(samples))