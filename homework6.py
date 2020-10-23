# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv("POP.csv")


# %%
def my_mean(m):
    total = 0.0
    for x in m:
        total = total + float(x)
    return float(total)/len(m)


# %%
def my_w1(X, Y):
    x_mean = my_mean(X)
    y_mean = my_mean(Y)
    num = 0.0
    den = 0.0
    for i in  range(len(X)):
        num = num + (X[i]- x_mean) * (Y[i] - y_mean)
        den = den + (X[i]- x_mean)*(X[i]- x_mean)
    return num/den


# %%
def my_w0(X, Y):
    mean_y = my_mean(Y)
    mean_x = my_mean(X)
    return mean_y - my_w1(X, Y)*mean_x


# %%
Y = data['value']

X = list(range(Y.size))
print(X)
print(Y)


# %%
# print(X)
print(type(y))
plot.plot(X, Y, 'o')
plot.xlabel('Index')
plot.ylabel('Value')


# %%
my_w1(X, Y)


# %%
my_w0(X, Y)


# %%
w1 = my_w1(X, Y)
w0 = my_w0(X, Y)
y_estimated= []
for x in X:
    y = w1 * x + w0
    y_estimated.append(y)


# %%
y_estimated


# %%
x_samples = np.linspace(0, 1000, 1000)
y_samples = []
for x in x_samples:
    y_samples.append(w0+w1*x)


# %%
plot.plot(X, Y, 'o')
plot.plot(x_samples, y_samples, color='green')
plot.xlabel('Index')
plot.ylabel('Value')


# %%
plot.axis([0, 7, 0, 7])
plot.plot(X, Y, 'o')
plot.plot(x_samples, y_samples, color='green')
for i in range(len(X)):
    x_pair = [X[i], X[i]]
    y_pair = [Y[i], y_estimated[i]]
    plot.plot(x_pair, y_pair, color='red')
plot.xlabel('Index')
plot.ylabel('Value')


# %%
S = [3, 8, 9, 13, 3, 6, 11, 21, 1, 16]
T = [30, 57, 64, 72, 36, 43, 59, 90, 20, 83]
w1 = my_w1(S, T)
w0 = my_w0(S, T)


