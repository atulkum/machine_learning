## Reprametrization trick for stochastic gradient descent

1) related to coupling
In probability theory, coupling is a proof technique that allows one 
to compare two unrelated random variables(distributions) 
{\displaystyle X} X and {\displaystyle Y} Y by creating a random vector 
{\displaystyle W} W whose marginal distributions correspond to {\displaystyle X} X and 
{\displaystyle Y} Y respectively. The choice of {\displaystyle W} W is generally not unique, and 
the whole idea of "coupling" is about making such a choice so that {\displaystyle X} X and {\displaystyle Y} 
Y can be related in a certain way we desire.

2) For gaussian random variable its simple (we can easily related x and y using a linear equation)

Gaussian var X, Y
X ~ N(0, 1)
each sample x and y from (X,Y) respectively are related
y = x + mu

so if we sample x_hat from x then we can use x_hat as a proxy for y_hat (sample from y) 

infinitesimal chnage in mu could be back propagated
dy/dmu = 1 (as x is independent of mu)
