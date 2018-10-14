demo(Chapter.4, package="mcsm", ask=T)

Nsim=5000
n=15
a=3
b=7
#initial values
#X=T=array(0,dim=c(Nsim,1))
X <- vector(length=Nsim)
T <- vector(length=Nsim)
#init arrays
T[1]=rbeta(1,a,b)
#init chains
X[1]=rbinom(1,n,T[1])
for (i in 2:Nsim){
  X[i]=rbinom(1,n,T[i-1])
  T[i]=rbeta(1,a+X[i],n-X[i]+b) 
}
fx <- dbeta(T, a,b)
f <- fx; f[fx == Inf] <- 1e100
f1 <- function(x){
  fx <- dbeta(T, a,b)
  f <- fx
  f[fx == Inf] <- 1e100
  f
}
hist(T,freq=F,col="wheat2",main="")
curve(dbeta(x, a,b), add=TRUE,col="tomato")

betai <- function(x,a,b,n){
  b1 <- beta(a,b)
  b2 <- beta(x+a, n-x+b)
  #ch <- choose(n, x) 
  ch1 <- gamma(n+1)/(gamma(x+1)*gamma(n-x+1))
  (ch1*b2)/b1
}
hist(X,freq=F,col="wheat2",main="")
curve(betai(x,a,b,n), add=TRUE,col="tomato")


x=c(91,504,557,609,693,727,764,803,857,929,970,1043,1089,1195,1384,1713)
Nsim=50000
a =3
b = 3
tau2 = 10
theta0 = 5
xbar=mean(x)
sh1=(n/2)+a
sigma2=theta=rep(0,Nsim)
sigma2[1]=1/rgamma(1,shape=a,rate=b)
B=sigma2[1]/(sigma2[1]+n*tau2)
theta[1]=rnorm(1,m=B*theta0+(1-B)*xbar,sd=sqrt(tau2*B))
for (i in 2:Nsim){
   B=sigma2[i-1]/(sigma2[i-1]+n*tau2)
   theta[i]=rnorm(1,m=B*theta0+(1-B)*xbar,sd=sqrt(tau2*B))
   ra1=(1/2)*(sum((x-theta[i])^2))+b
   sigma2[i]=1/rgamma(1,shape=sh1,rate=ra1) 
}
hist(log(sigma2),freq=F,col="wheat2",main="")
hist(log(theta),freq=F,col="wheat2",main="")

for(i in 2:Nsim){
  zbar[i]=mean(rtrun(mean=rep(that[i-1],n-m),
                       sigma=rep(1,n-m),a=rep(a,n-m),b=rep(Inf,n-m)))
  that[i]=rnorm(1,(m/n)*xbar+(1-m/n)*zbar[i],sqrt(1/n)) 
}

x=c(125,18,20,34)             #data
theta=z=rep(.5,Nsim)          #init chain
for (j in 2:Nsim){
    theta[j]=rbeta(1,z[j-1]+x[4]+1,x[2]+x[3]+1)
    z[j]=rbinom(1,x{1},(theta[j]/(2+theta[j]))) 
}