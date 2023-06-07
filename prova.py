import numpy as np
import matplotlib.pyplot as plt 
vector = [("theta",2.0,0)]*5 + [("gamma",1.0,0)]*5 + [("normal",0,4.0)]*(5)
np.random.shuffle(vector)
vector[1]
x = np.asarray([1.0, np.nan, 3.0])
x[~np.isnan(x)]
Y = np.array([[0,np.nan,0],[1,0,0]])
x[:, np.newaxis]
Y*x
np.nanprod(x)
print(x)


matrix = np.array([[1, np.nan, 3],
                   [np.nan, 5, 6],
                   [7,np.nan, np.nan]])

# Drop 'na' values from the matrix
matrix_dropped = np.array([row[~np.isnan(row.astype(float))] for row in matrix])

print(matrix_dropped)