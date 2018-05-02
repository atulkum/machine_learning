NSims = 10^4
rho = 0.9
x <- vector(length=NSims)
#x=1:10^4
x[1] = rnorm(1)
for (i in 2:NSims){
  x[i] = rho*x[i-1] + rnorm(1)
}
hist(x,freq=F,col="wheat2",main="")
curve(dnorm(x,sd=1/sqrt(1-rho^2)),add=T,col="tomato")
      

NSims = 10^6
rho = 0.9
x <- vector(length=NSims)
#x=1:10^4
x[1] = 0 #rnorm(1)
for (i in 2:NSims){
  x[i] = rho*x[i-1] + rnorm(1)
}

hist(x,freq=F,col="wheat2",main="")
hist(x[1:10^4],freq=F,col="tomato",main="")


y=geneq(x[t])
rho = min(f(y)*q(y,x[t])/(f(x[t])*q(x[t],y)), 1)
if (runif(1)<rho){
  x[t+1]=y
}else{
  x[t+1]=x[t]
}

a=2.7; b=6.3; c=2.669 # initial values
Nsim=5000
X=rep(runif(1),Nsim)  # initialize the chain
for (i in 2:Nsim){
  Y=runif(1)
  rho=dbeta(Y,a,b)/dbeta(X[i-1],a,b)
  X[i]=X[i-1] + (Y-X[i-1])*(runif(1)<rho) 
}

plot(1:5000,X, type="l")
plot(4500:4800,X[4500:4800], type="l")

ks.test(jitter(X),rbeta(5000,a,b))

mu = a/(a+b)
var = (a*b)/((a + b)^2*(a + b + 1))

mean(X)
var(X)

h <- function(x){
  dbeta(x,2.7,6.3)
}

hist(X,freq=F,col="wheat2",main="")
lines(density(X),col="tomato")

X_iid = rbeta(Nsim,a,b)
hist(X_iid,freq=F,col="wheat2",main="")
lines(density(X_iid),col="tomato")





