import numpy as np
import matplotlib.pyplot as plt

#genrate sequence of normal random variables
mu = 0
sigma = np.sqrt(0.015)
size = 5000
x = np.random.normal(mu, sigma, size)
theta = np.random.normal(mu, sigma, size)

print("variance of 2x is:", np.var(2*x), "it should be: ", 4*0.015)
#generate the lognormal random variable

y = np.exp(x)
z = np.exp(theta)
epsilon = np.exp(x+x)
x2 = np.exp(x+theta)

print("mean of epsilon is: ", np.mean(epsilon), "mean of x2 is: ", np.mean(x2))
#compute covariance of yz and y
cov = np.cov(y*z,y)
print("covariance is: ", cov[0,1], "it should be: ", np.exp((5*0.015)/2) - np.exp((3*0.015)/2) )

#compute and print mean and variance of y

print("mean is: ", np.mean(y), "theoretical mean is: ", np.exp(mu + sigma**2/2))
print("variance is: ", np.var(y), "theoretical variance is: ", (np.exp(sigma**2) - 1)*np.exp(2*mu + sigma**2))
g = np.exp(mu + sigma**2/2) - 1
print("growth rate is: ", g)
R = 1.02
print("upper bound is: ", np.log(R))
#define contsant growth dividend process
d = np.ones(size)
d = d * np.cumprod(y)
p = d*(1 + g)/(R-1-g)
#plot the price process
plt.plot(p)


#Check first order autocorrelation of price
ac = np.corrcoef(p[1:], p[:-1])
print("first order autocorrelation of price is: ", ac[0,1])

#compute returns
ret = np.log(p[1:]/p[:-1])

#plot returns
fix, ax = plt.subplots()
ax.hist(ret, bins = 30, density = True, label = "returns")
#compare with normal distribution with same mean and variance
ret_mu = np.mean(ret)
ret_sigma = np.std(ret)
x = np.linspace(ret_mu - 3*ret_sigma, ret_mu + 3*ret_sigma, 100)
ax.plot(x, 1/(ret_sigma*np.sqrt(2*np.pi))*np.exp(-(x - ret_mu)**2/(2*ret_sigma**2)), label = "normal distribution")

#plot returns
fig1, ax1 = plt.subplots()
ax1.plot(ret)

plt.show()