import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def prob(x, v = 0.2):
    return norm(loc = x , scale = np.sqrt(v)).cdf(0)

x = np.linspace(-5,5,50)
y = prob(x)
plt.plot(x,y)
plt.show()

    