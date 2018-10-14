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

Nsim=5000
#target = rgamma(Nsim, 4.85,1)
#candicate = dgamma(x,4, 4/4.85)
#accept -reject
M=optimize(f=function(x){dgamma(x, 4.85,1)/dgamma(x,4, 4/4.85)},maximum=T,interval=c(0,1))$objective
uM =runif(Nsim,max=M)
y = rgamma(Nsim, 4, 4/4.85)
fy = dgamma(y, 4.85,1)
gy = dgamma(y,4, 4/4.85)
uM.gy = uM*gy
accep2 = uM.gy < fy


accept1=rep(F, Nsim)
accept1[1] = T
x = 1:Nsim
x[1] = y[1]
for (i in 2:Nsim){
  rho= (dgamma(y[i], 4.85,1)*dgamma(x[i-1],4, 4/4.85))/(dgamma(x[i-1], 4.85,1)*dgamma(y[i],4, 4/4.85))
  rho = min(rho, 1)
  accept1[i] = (runif(1)<rho)
  if(accept1[i]){
    x[i] = y[i]
  }else{
    x[i] = x[i-1]
  }
}

####
pnorm(12.78,log=T,low=F)/log(10)
pt(3,1)

Nsim=10
X_norm = 1:Nsim
X_norm[1]=c(rt(1,1))    # initialize the chain from the stationary
for (t in 2:Nsim){
    #Y= rnorm(1)   # candidate normal 
    #if(dnorm(Y) == 0){
      #rho=dt(Y,1)*dnorm(X_norm[t-1])/(dt(X_norm[t-1],1)*dnorm(Y))
    Y= rt(1,.5)
    gy = dt(Y, 1.5)
    fx = dt(X_norm[t-1],1)
    if((gy == 0)){
      rho = 1
    }else{
      rho=dt(Y,1)*dt(X_norm[t-1], 1.5)/(fx*fy)
    }
    rho = min(rho, 1)
    X_norm[t]=X[t-1] + (Y-X_norm[t-1])*(runif(1)<rho) 
}
plot(cumsum(X_norm<3)/(1:Nsim),lwd=2,ty="l",ylim=c(.85,1))

x = cars$speed
y = cars$dist

x2=x^2
summary(lm(y~x+x2))

a ∼ N (2.47, (14.8)^2 ), 
b ∼ N (.913, (2.03)^2 ), c ∼ N (.099, (0.065)^2 ), 
n = 50
σ−2 ∼ Gamma(n/2, (n − 3)(15.18)^2),

(Intercept)  2.47014   14.81716   0.167    0.868
x            0.91329    2.03422   0.449    0.656
x2           0.09996    0.06597   1.515    0.136

Residual standard error: 15.18 on 47 degrees of freedom
Multiple R-squared:  0.6673,  Adjusted R-squared:  0.6532 
F-statistic: 47.14 on 2 and 47 DF,  p-value: 5.852e-12

da=rbind(rnorm(10^2),2.5+rnorm(3*10^2))
like=function(mu){
     sum(log((.25*dnorm(da-mu[1])+.75*dnorm(da-mu[2]))))}

sta=c(1,1)
mmu=sta
for (i in 1:(nlm(like,sta)$it))
     mmu=rbind(mmu,nlm(like,sta,iterlim=i)$est)
plot(mmu,pch=19,lwd=2, type="l")

scale=1
the=matrix(runif(2,-2,5),ncol=2)
curlike=hval=like(the)
Niter=10^4
for (iter in (1:Niter)){
  prop=the[iter,]+rnorm(2)*scale
  if ((max(-prop)>2)||(max(prop)>5)||
        (log(runif(1))>like(prop)-curlike)) prop=the[iter,]
  
  curlike=like(prop)
  hval=c(hval,curlike)
  the=rbind(the,prop)
}

library(MASS)
like=function(beda){
  mia=mean(Pima.tr$bmi)
  prod(pnorm(beda[1]+(Pima.tr$bm[Pima.tr$t=="Yes"]-
                        mia)*beda[2]))*
    prod(pnorm(-beda[1]-(Pima.tr$bm[Pima.tr$t=="No"]
                         -mia)*beda[2]))/exp(sum(beda^2)/200)
}
grad=function(a,b){
  don=pnorm(q=a+outer(X=b,Y=da[,2],FUN="*"))
  x1=sum((dnorm(x=a+outer(X=b,Y=da[,2],FUN="*"))/don)*da[,1]-
           (dnorm(x=-a-outer(X=b,Y=da[,2],FUN="*"))/
              (1-don))*(1-da[,1]))
  x2=sum(da[,2]*(
    (dnorm(x=a+outer(X=b,Y=da[,2],FUN="*"))/don)*da[,1]-
      (dnorm(x=-a-outer(X=b,Y=da[,2],FUN="*"))/
         (1-don))*(1-da[,1])))
  return(c(x1,x2))
}
prop=curmean+scale*rnorm(2)
propmean=prop+0.5*scale^2*grad(prop[1],prop[2])
if (log(runif(1))>like(prop[1],prop[2])-likecur-
      sum(dnorm(prop,mean=curmean,sd=scale,lo=T))+
      sum(dnorm(the[t-1,],mean=propmean,sd=scale,lo=T))){
  prop=the[t-1,];propmean=curmean
}

gradlike=function(mu){
  deno=.2*dnorm(da-mu[1])+.8*dnorm(da-mu[2])
  gra=sum(.2*(da-mu[1])*dnorm(da-mu[1])/deno)
  grb=sum(.8*(da-mu[2])*dnorm(da-mu[2])/deno)
  return(c(gra,grb))
}

prop=curmean+rnorm(2)*scale
meanprop=prop+.5*scale^2*gradlike(prop)
if ((max(-prop)>2)||(max(prop)>5)||(log(runif(1))>like(prop)-curlike-
        sum(dnorm(prop,curmean,lo=T))+
        sum(dnorm(the[iter,],meanprop,lo=T)))){
  prop=the[iter,]
  meanprop=curmean 
}
curlike=like(prop)
curmean=meanprop

names(swiss)
y=log(as.vector(swiss[,1]))
X=as.matrix(swiss[,2:6])

inv=function(X){
  EV=eigen(X)
  EV$vector%*%diag(1/EV$values)%*%t(EV$vector)
}
lpostw=function(gam,y,X,beta){
  n=length(y)
  qgam=sum(gam)
  Xt1=cbind(rep(1,n),X[,which(gam==1)])
  if (qgam!=0) P1=Xt1%*%inv(t(Xt1)%*%Xt1)%*%t(Xt1) 
  else{
    P1=matrix(0,n,n)
  }
  -(qgam+1)/2*log(n+1)-n/2*log(t(y)%*%y-n/(n+1)*
  t(y)%*%P1%*%y-1/(n+1)*t(beta)%*%t(cbind(rep(1,n),X))%*%P1%*%cbind(rep(1,n),X)%*%beta)
}


gocho=function(niter,y,X){ 
  lga=dim(X)[2]                         
  beta=lm(y∼X)$coeff 
  gamma=matrix(0,nrow=niter,ncol=lga) 
  gamma[1,]=sample(c(0,1),lga,rep=T) 
                         
  for (t in 1:(niter-1)){ 
    j=sample(1:lga,1)    
    gam0=gam1=gamma[t,];gam1[j]=1-gam0[j]        
    pr=lpostw(gam0,y,X,beta)
    pr=c(pr,lpostw(gam1,y,X,beta)) 
    pr=exp(pr-max(pr))
    gamma[t+1,]=gam0       
    if (sample(c(0,1),1,prob=pr)) gamma[t+1,]=gam1
  } 
  gamma
}
out=gocho(10^5,y,X)
apply(out,2,mean)