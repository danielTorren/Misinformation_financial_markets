import numpy as np
import matplotlib.pyplot as plt
data = np.cumsum(np.random.normal(0, 1, 1000) + 0.9 * np.concatenate(([0], data[:-1])))
plt.plot(data)

def generate_ar1(mean, acf, sigma, N):
    data = [mean]
    for i in range(1,N):
        noise = np.random.normal(0,sigma)
        data.append(acf * data[-1] + noise)
    return np.array(data)


# vector = [("theta",2.0,0)]*5 + [("gamma",1.0,0)]*5 + [("normal",0,4.0)]*(5)
# np.random.shuffle(vector)
# vector[1]
# x = np.asarray([1.0, np.nan, 3.0])
# x[~np.isnan(x)]
# Y = np.array([[0,np.nan,0],[1,0,0]])
# x[:, np.newaxis]
# Y*x
# np.nanprod(x)
# print(x)


# matrix = np.array([[1, np.nan, 3],
#                    [np.nan, 5, 6],
#                    [7,np.nan, np.nan]])

# # Drop 'na' values from the matrix
# matrix_dropped = np.array([row[~np.isnan(row.astype(float))] for row in matrix])

# print(matrix_dropped)