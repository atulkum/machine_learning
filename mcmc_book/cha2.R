rgamma(3,2.5,4.5)

runif(100, min=2, max=5)

Nsim=10^4
x=runif(Nsim)
x1=x[-Nsim]
x2=x[-1]
par(mfrow=c(1,3))
hist(x)
plot(x1,x2)
acf(x)

set.seed(1)
runif(5)

set.seed(2)
runif(5)

Nsim=10^4
U=runif(Nsim)
X=-log(U)
Y=rexp(Nsim)
par(mfrow=c(1,2))
hist(X,freq=F,main="Exp from Uniform")
hist(Y,freq=F,main="Exp from R")

test1 <- function(){
  U=runif(3*10^4)
  U=matrix(data=U,nrow=3) #matrix for sums
  X=-log(U) #uniform to exponential
  X=2* apply(X,2,sum) 
}
test2 <- function(){
  X=rchisq(10^4,df=6)
}

par(mfrow=c(1,2))
hist(test1(),freq=F,main="Exp from Uniform")
hist(test2(),freq=F,main="Exp from R")

system.time(test1());system.time(test2())


box_muller <- function(){
  U1=runif(10^4)
  U2=runif(10^4)
  U2=-log(U2)
  X1 = sqrt(-2*log(U1))*cos(2*pi*U2) 
  X2 = sqrt(-2*log(U1))*sin(2*pi*U2) 
  
  return (list('X1' = X1, 'X2' = X2))
}
ret = box_muller()

#par(mfrow=c(1,2))
#hist(ret$X1,freq=F,main="X1")
#hist(ret$X2,freq=F,main="X2")

clt12 <- function(){
  U=runif(12*10^4, -1/2, 1/2)
  U=matrix(data=U,nrow=12) 
  X=apply(U,2,sum) 
  return (X)
}
X_clt = clt12()

X_r = rnorm(12*10^4)

par(mfrow=c(1,3))
hist(ret$X1,freq=F,main="box_muller")
hist(X_clt,freq=F,main="clt")
hist(X_r,freq=F,main="r")

library(matrixcalc)
library(Matrix) 

sampleMultiNorm <- function(A, d, Nsim){
  for(ni in 1:Nsim){
    x=A%*%rnorm(d)
  }
}
library(mnormt)
sampleRnorm <- function(A, d, Nsim){
  rmnorm(Nsim,varcov=A)
}

Nsim=10^4; d = 100
Sigma=cov(matrix(rnorm(d*Nsim),nrow=Nsim))
#is.positive.definite(Sigma)
#is.symmetric.matrix(S)
A=t(chol(Sigma))

system.time(sampleMultiNorm(A, d, Nsim));system.time(sampleRnorm(A, d, Nsim))

discretePois <- function(Nsim, lambda){
  spread=3*sqrt(lambda)
  t=round(seq(max(0,lambda-spread),lambda+spread,1))
  prob=ppois(t, lambda)
  X=rep(0,Nsim)
  for (i in 1:Nsim){
    u=runif(1)
    X[i]=t[1]+sum(prob<u) 
  }
}
Nsim=10^4; lambda=100
system.time(discretePois(Nsim, lambda)); system.time(ppois(Nsim, lambda))


Nsim=10^4
n=6;p=.3
y=rgamma(Nsim,n,rate=p/(1-p))
x=rpois(Nsim,y)
hist(x,main="",freq=F,col="grey",breaks=40)
lines(1:50,dnbinom(1:50,n,p),lwd=2,col="sienna")

accept_reject_single <- function(M, f, g){
  u=runif(1)*M
  y=randg(1)
  while (u > f(y)/g(y)){
    u=runif(1)*M
    y=randg(1)
  }
  return y
}
accept_reject <- function(M, f, g, Nsim){
  u=runif(Nsim)*M
  y=randg(Nsim)
  return y[u < f(y)/g(y)]
}


Nsim=2500
a=2.7;b=6.3

gendbeta <- function(Nsim, a, b){
  M=optimize(f=function(x){dbeta(x,a,b)}, interval=c(0,1),maximum=T)$objective
  u=runif(Nsim,max=M)
  y=runif(Nsim)
  #dunif(y) == 1
  fy = dbeta(y,a,b)
  accep = u < fy
  x=y[accep]

  pltx = seq(0, 1, length=100)
  plot(pltx,dbeta(pltx,a,b),lwd=2,col="sienna", xlab="y", ylab="u.g(y)")
  lines(pltx,rep(M, length(pltx)),lwd=2,col="green")
  
  points(y, u, col = "grey")
  points(x, u[accep], col = "black")
  print (length(x)/length(y))
}
gendbeta(Nsim, a, b)

a=2;b=6
gendbeta(Nsim, a, b)

x=NULL
while (length(x)<Nsim){
  y=runif(Nsim*M)
  x=c(x,y[runif(Nsim*M)*M<dbeta(y,a,b)])
}
x=x[1:Nsim]



M=optimize(f=function(x){dbeta(x,2.7,6.3)/dbeta(x,2,6)},maximum=T,interval=c(0,1))$objective

uM =runif(Nsim,max=M)
y = rbeta(Nsim,2,6)
fy = dbeta(y,2.7,6.3)
gy = dbeta(y,2,6)
uM.gy = uM*gy
accep = uM.gy < fy
x=y[accep]

pltx = seq(0, 0.8, length=100)
plot(pltx,dbeta(pltx,2,6)*M,type="l", lwd=2,col="green", xlab="y", ylab="u.g(y)")
lines(pltx,dbeta(pltx,2.7,6.3),lwd=2,col="sienna")
points(y, uM.gy, col = "grey")
points(x, uM.gy[accep], col = "black")
print (length(x)/length(y))
