
# coding: utf-8

# In[4]:


import math 
import pandas 
import numpy as np 
import scipy.stats 
from scipy.stats import uniform
from scipy.stats import binom
import math
from scipy.stats import norm
import matplotlib    
import matplotlib.pyplot as plt


# In[8]:


#Q.1 from HW2
n = 10
p = 0.3
k = np.arange(0,21)
binomial = scipy.stats.binom.pmf(k,n,p)
print(binomial)
plt.plot(k,binomial,'o-')
plt.title('Binomial: n=%i, p=%.2f' % (n,p), fontsize = 15)
plt.xlabel('Number of Successes')
plt.ylabel('Probability of Successes', fontsize=15)
plt.show()


# In[11]:


#Q.2 from HW2
def box_muller():
    u1 = random.random()
    u2 = random.random()

    t = math.sqrt((-2) * math.log(u1))
    v = 2 * math.pi * u2

    return t * math.cos(v), t * math.sin(v)


# In[16]:


from numpy import random, sqrt, log, sin, cos, pi
from pylab import show,hist,subplot,figure

# transformation function
def gaussian(u1,u2):
  z1 = sqrt(-2*log(u1))*cos(2*pi*u2)
  z2 = sqrt(-2*log(u1))*sin(2*pi*u2)
  return z1,z2

# uniformly distributed values between 0 and 1
u1 = random.rand(1000)
u2 = random.rand(1000)

# run the transformation
z1,z2 = gaussian(u1,u2)

# plotting the values before and after the transformation
figure()
subplot(221) # the first row of graphs
hist(u1)     # contains the histograms of u1 and u2 
subplot(222)
hist(u2)
subplot(223) # the second contains
hist(z1)     # the histograms of z1 and z2
subplot(224)
hist(z2)
show()
#In the first row of the graph we can see, respectively, the histograms of u1 and u2 before the transformation 
#and in the second row we can see the values after the transformation, respectively z1 and z2. 
#We can observe that the values before the transformation are distributed uniformly while the histograms of the values 
#after the transformation have the typical Gaussian shape.
#The Box-Muller transform is a method for generating normally distributed random numbers from uniformly distributed 
#random numbers. The Box-Muller transformation can be summarized as follows, suppose u1 and u2 are independent random variables
#that are uniformly distributed between 0 and 1 and let 
#then z1 and z2 are independent random variables with a standard normal distribution. Intuitively, the transformation maps 
#each circle of points around the origin to another circle of points around the origin where larger outer circles are mapped 
#to closely-spaced inner circles and inner circles to outer circles. 


# In[21]:


import scipy.stats as ss

n = 15         # Number of total bets
p = 0.3     # Probability of getting "red" at the roulette
max_sbets = 0  # Maximum number of successful bets

hh = ss.binom(n, p)

total_p = 0
for k in range(1, max_sbets + 1):  # DO NOT FORGET THAT THE LAST INDEX IS NOT USED
    total_p += hh.pmf(k)


# In[22]:


total_p


# In[23]:


import scipy.stats as ss

n = 15         # Number of total bets
p = 0.2     # Probability of getting "red" at the roulette
max_sbets = 1  # Maximum number of successful bets

hh = ss.binom(n, p)

total_p = 0
for k in range(1, max_sbets + 1):  # DO NOT FORGET THAT THE LAST INDEX IS NOT USED
    total_p += hh.pmf(k)
total_p


# In[24]:


import scipy.stats as ss

n = 15         # Number of total bets
p = 0.5     # Probability of getting "red" at the roulette
max_sbets = 3  # Maximum number of successful bets

hh = ss.binom(n, p)

total_p = 0
for k in range(1, max_sbets + 1):  # DO NOT FORGET THAT THE LAST INDEX IS NOT USED
    total_p += hh.pmf(k)
total_p


# In[25]:


#! /usr/local/bin/python3.6
"""
Random number generatrion with Box-Muller algorithm
"""
import math
import random
import sys
import traceback


class RndnumBoxMuller:
    M     = 10        # Average
    S     = 2.5       # Standard deviation
    N     = 10000     # Number to generate
    SCALE = N // 100  # Scale for histgram

    def __init__(self):
        self.hist = [0 for _ in range(self.M * 5)]

    def generate_rndnum(self):
        """ Generation of random numbers """
        try:
            for _ in range(self.N):
                res = self.__rnd()
                self.hist[res[0]] += 1
                self.hist[res[1]] += 1
        except Exception as e:
            raise

    def display(self):
        """ Display """
        try:
            for i in range(0, self.M * 2 + 1):
                print("{:>3}:{:>4} | ".format(i, self.hist[i]), end="")
                for j in range(1, self.hist[i] // self.SCALE + 1):
                    print("*", end="")
                print()
        except Exception as e:
            raise

    def __rnd(self):
        """ Generation of random integers """
        try:
            r_1 = random.random()
            r_2 = random.random()
            x = self.S               * math.sqrt(-2 * math.log(r_1))               * math.cos(2 * math.pi * r_2)               + self.M
            y = self.S               * math.sqrt(-2 * math.log(r_1))               * math.sin(2 * math.pi * r_2)               + self.M
            return [math.floor(x), math.floor(y)]
        except Exception as e:
            raise


if __name__ == '__main__':
    try:
        obj = RndnumBoxMuller()
        obj.generate_rndnum()
        obj.display()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


# In[26]:


#!/usr/bin/env python
#Box-Muller method
#to generate gaussian values from the numbers distributed uniformly.

import numpy as np
import matplotlib.pyplot as plt

#generate from uniform dist
np.random.seed()
N = 1000
z1 = np.random.uniform(0, 1.0 ,N)
z2 = np.random.uniform(0, 1.0 ,N)
z1 = 2*z1 - 1
z2 = 2*z2 - 1

#discard if z1**2 + z2**2 <= 1
c = z1**2 + z2**2
index = np.where(c<=1)
z1 = z1[index]
z2 = z2[index]
r = c[index]

#transformation
y1 = z1*((-2*np.log(r**2))/r**2)**(0.5)
y2 = z2*((-2*np.log(r**2))/r**2)**(0.5)

#discard outlier
y1 = y1[y1 <= 5]
y1 = y1[y1 >= -5]
y2 = y2[y2 <= 5]
y2 = y2[y2 >= -5]

#plot
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.hist(y1,bins=30,color='red')
plt.title("Histgram")
plt.xlabel("y1")
plt.ylabel("frequency")
ax2 = fig.add_subplot(2,1,2)
ax2.hist(y2,bins=30,color='blue')
plt.xlabel("y2")
plt.ylabel("frequency")
plt.show()


# In[28]:


a=3
b=3
x=np.arange(0,1,0.1)
y=scipy.stats.beta.pdf(x,a,b)
print(y)
plt.plot(x,y)


# In[73]:


#Problem 4-Q.3.2
import math
N = 1000
U = np.random.uniform(N)
print(U)
x = math.log((-1)*(1-U)/2)
print(x)
plt.plot(x)
hist(x)
#hist(X, freq=F, xlab='X', main='Generating Exponential R.V.')
#curve(dexp(x, rate=2) , 0, 3, lwd=2, xlab = "", ylab = "", add = T)
#plot
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.hist(x,bins=30,color='red')
plt.title("X")
plt.xlabel("x")
plt.ylabel("U")
ax2 = fig.add_subplot(2,1,2)
ax2.hist(U,bins=30,color='blue')
plt.xlabel("U")
plt.ylabel("U")
plt.show()




# In[104]:


import math 
import pandas 
import numpy as np 
import scipy.stats 
from scipy.stats import uniform
import math
from scipy.stats import norm
import matplotlib    
import matplotlib.pyplot as plt

x=np.random.beta(1000,3,2)
y =scipy.stats.beta.pdf(1000,3,2)
print(x,y)
hist(x)
hist(y)
plt.plot(x)
plt.plot(y)


# In[83]:


np.random.beta(1000,3,2)


# In[ ]:




