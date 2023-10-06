import numpy as np
import matplotlib.pyplot as plt

def generate_ar1(mean, acf, sigma, N):
    data = [mean]
    for i in range(1,N):
        noise = np.random.normal(0,sigma)
        data.append(acf * data[-1] + noise)
    return np.array(data)

x = generate_ar1(0, 0.8, 1, 10000)
print(np.var(x))
