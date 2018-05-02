library(MASS)

ch <- function(la){
  integrate(function(x){x^(la-1)*exp(-x)},0,Inf)$val
}
x = seq(.01,10,le=100)

plot(lgamma(x),
     log(apply(as.matrix(x),1,ch)),
     xlab="log(integrate(f))",
     ylab=expression(log(Gamma(lambda))),pch=19,cex=.6)


cac=rcauchy(10)+350

lik=function(the){
  u=dcauchy(cac[1]-the)
  for (i in 2:10)
    u=u*dcauchy(cac[i]-the)
  return(u)
}
 
integrate(lik,-Inf,Inf)
integrate(lik,200,400)

cac=rcauchy(10)
nin=function(a){integrate(lik,-a,a)$val}
nan=function(a){area(lik,-a,a)}
x=seq(1,10^3,le=10^4)
y=log(apply(as.matrix(x),1,nin))
z=log(apply(as.matrix(x),1,nan))
plot(x,y,type="l",ylim=range(cbind(y,z)),lwd=2)
lines(x,z,lty=2,col="sienna",lwd=2)

#####calculating cauchy mle
mlecauchy=function(x,toler=.001){      #x is a vector here
  startvalue=median(x)
  n=length(x);
  thetahatcurr=startvalue;
  # Compute first deriviative of log likelihood
  firstderivll=2*sum((x-thetahatcurr)/(1+(x-thetahatcurr)^2))
  # Continue Newton’s method until the first derivative
  # of the likelihood is within toler of 0.001
  while(abs(firstderivll)>toler){
    # Compute second derivative of log likelihood
    secondderivll=2*sum(((x-thetahatcurr)^2-1)/(1+(x-thetahatcurr)^2)^2);
    # Newton’s method update of estimate of theta
    thetahatnew=thetahatcurr-firstderivll/secondderivll;
    thetahatcurr=thetahatnew;
    # Compute first derivative of log likelihood
    firstderivll=2*sum((x-thetahatcurr)/(1+(x-thetahatcurr)^2))
  }
  list(thetahat=thetahatcurr);
}
x<-c(-1.94,0.59,-5.98,-0.08,-0.77)
mlecauchy(x,0.0001)
obj = optimize(function(theta) -sum(dcauchy(x, location=theta, log=TRUE)),  c(-100,100)) 
obj
##########

h=function(x){(cos(50*x)+sin(20*x))^2}
par(mar=c(2,2,2,1),mfrow=c(2,1))
curve(h,xlab="Function",ylab="",lwd=2)
integrate(h,0,1)

x=h(runif(10^4))
estint=cumsum(x)/(1:10^4)
esterr=sqrt(cumsum((x-estint)^2))/(1:10^4)
plot(estint, xlab="Mean and error range",type="l",lwd=
         2,ylim=mean(x)+20*c(-esterr[10^4],esterr[10^4]),ylab="")
lines(estint+2*esterr,col="gold",lwd=2)
lines(estint-2*esterr,col="gold",lwd=2)


integrand1 <- function(theta, x){
  (theta/(1+theta^2))*exp(-((x-theta)^2)/2)
}
integrand2 <- function(theta, x){
  (1/(1+theta^2))*exp(-((x-theta)^2)/2)
}

x = 2
plot(integrand1(,x),type="l",lwd=2,ylab="")

x=rnorm(10^8)                #whole sample
bound=qnorm(c(.5,.75,.8,.9,.95,.99,.999,.9999))
res=matrix(0,ncol=8,nrow=7)
for (i in 2:8)               #lengthy loop!!
  for (j in 1:8)
      res[i-1,j]=mean(x[1:10^i]<bound[j])
matrix(as.numeric(format(res,digi=4)),ncol=8)

Nsim=10^3
y=rexp(Nsim)+4.5
weit=dnorm(y)/dexp(y-4.5)
plot(cumsum(weit)/1:Nsim,type="l")
abline(a=pnorm(-4.5),b=0,col="red")

f=function(a,b){
  exp(2*(lgamma(a+b)-lgamma(a)-lgamma(b))+ a*log(.3)+b*log(.2))
}

aa=1:150      #alpha grid for image
bb=1:100      #beta grid for image
post=outer(aa,bb,f)
image(aa,bb,post,xlab=expression(alpha),ylab=" ")
contour(aa,bb,post,add=T)

x=matrix(rt(2*10^4,3),ncol=2)       #T sample
E=matrix(c(220,190,190,180),ncol=2) #Scale matrix
image(aa,bb,post,xlab=expression(alpha),ylab=" ")
y=t(t(chol(E))%*%t(x)+c(50,45))
points(y,cex=.6,pch=19)

ine=apply(y,1,min)
y=y[ine>0,]
x=x[ine>0,]
normx=sqrt(x[,1]^2+x[,2]^2)
f=function(a) exp(2*(lgamma(a[,1]+a[,2])-lgamma(a[,1])
                          -lgamma(a[,2]))+a[,1]*log(.3)+a[,2]*log(.2))
h=function(a) exp(1*(lgamma(a[,1]+a[,2])-lgamma(a[,1])
                          -lgamma(a[,2]))+a[,1]*log(.5)+a[,2]*log(.5))
den=dt(normx,3)
mean(f(y)/den)/mean(h(y)/den)

mean(y[,1]*apply(y,1,f)/den)/mean(apply(y,1,h)/den)
mean(y[,2]*apply(y,1,f)/den)/mean(apply(y,1,h)/den)

par(mfrow=c(2,2),mar=c(4,4,2,1))
weit=(apply(y,1,f)/den)/mean(apply(y,1,h)/den)
image(aa,bb,post,xlab=expression(alpha),
      ylab=expression(beta))
points(y[sample(1:length(weit),10^3,rep=T,pro=weit),],
    cex=.6,pch=19)
boxplot(weit,ylab="importance weight")
plot(cumsum(weit)/(1:length(weit)),type="l",
     xlab="simulations", ylab="marginal likelihood")
boot=matrix(0,ncol=length(weit),nrow=100)
for (t in 1:100)
   boot[t,]=cumsum(sample(weit))/(1:length(weit))
uppa=apply(boot,2,quantile,.95)
lowa=apply(boot,2,quantile,.05)
polygon(c(1:length(weit),length(weit):1),c(uppa,rev(lowa)),
         col="gold")
lines(cumsum(weit)/(1:length(weit)),lwd=2)
plot(cumsum(weit)^2/cumsum(weit^2),type="l",
     xlab="simulations", ylab="Effective sample size",lwd=2)

#ex 3.4
Nsim=10^3
h <- function(x){ exp(-(x - 3)*2/2) + exp(-(x - 6)*2/2)}

y = rnorm(Nsim)
x=h(y)
estint=cumsum(x)/(1:Nsim)
esterr=sqrt(cumsum((x-estint)^2))/(1:Nsim)
plot(estint, xlab="Mean and error range",type="l",lwd=
       2,ylim=mean(x)+20*c(-esterr[Nsim],esterr[Nsim]),ylab="")
lines(estint+2*esterr,col="gold",lwd=2)
lines(estint-2*esterr,col="gold",lwd=2)

Nsim = 10^9
y = runif(Nsim, -8, -1)
f_x = dnorm(y)
g_x = dunif(y, -8, -1) #1/10^3
h_x = h(y)
weit = f_x/g_x
x = weit*h_x
estint=cumsum(x)/(1:Nsim)
esterr=sqrt(cumsum((x-estint)^2))/(1:Nsim)
plot(estint, xlab="Mean and error range",type="l",lwd=
       2,ylim=mean(x)+20*c(-esterr[Nsim],esterr[Nsim]),ylab="")
lines(estint+2*esterr,col="gold",lwd=2)
lines(estint-2*esterr,col="gold",lwd=2)
#####

pnorm(-4.5,log=T)

Nsim=10^3
y=rexp(Nsim)+4.5
weit=dnorm(y)/dexp(y-4.5)
plot(cumsum(weit)/1:Nsim,type="l")
abline(a=pnorm(-4.5),b=0,col="red")

x = weit
estint=cumsum(x)/(1:Nsim)
esterr=sqrt(cumsum((x-estint)^2))/(1:Nsim)
plot(estint, xlab="Mean and error range",type="l",lwd=
       2,ylim=mean(x)+20*c(-esterr[Nsim],esterr[Nsim]),ylab="")
lines(estint+2*esterr,col="gold",lwd=2)
lines(estint-2*esterr,col="gold",lwd=2)
abline(a=pnorm(-4.5),b=0,col="red")


f=function(a,b){exp(2*(lgamma(a+b)-lgamma(a)-lgamma(b)) + a*log(.3)+b*log(.2))}
aa=1:150      #alpha grid for image
bb=1:100      #beta grid for image
post=outer(aa,bb,f)
image(aa,bb,post,xlab=expression(alpha),ylab=" ")
contour(aa,bb,post,add=T)

x=matrix(rt(2*10^4,3),ncol=2)       #T sample
E=matrix(c(220,190,190,180),ncol=2) #Scale matrix
#image(aa,bb,post,xlab=expression(alpha),ylab=" ")
y=t(t(chol(E))%*%t(x)+c(50,45))
#points(y,cex=.6,pch=19)

ine=apply(y,1,min)
y=y[ine>0,]
x=x[ine>0,]
normx=sqrt(x[,1]^2+x[,2]^2)
f<-function(a) {exp(2*(lgamma(a[,1]+a[,2])-lgamma(a[,1]) -lgamma(a[,2]))+a[,1]*log(.3)+a[,2]*log(.2))}
h<-function(a) {exp(1*(lgamma(a[,1]+a[,2])-lgamma(a[,1]) -lgamma(a[,2]))+a[,1]*log(.5)+a[,2]*log(.5))}
den=dt(normx,3)
mean(f(y)/den)/mean(h(y)/den)
mean(y[,1]*f(y)/den)/mean(h(y)/den)
mean(y[,2]*f(y)/den)/mean(h(y)/den)


par(mfrow=c(2,2),mar=c(4,4,2,1))
weit=(f(y)/den)/mean(h(y)/den)
image(aa,bb,post,xlab=expression(alpha),ylab=expression(beta))
points(y[sample(1:length(weit),10^3,rep=T,pro=weit),], cex=.6,pch=19)
boxplot(weit,ylab="importance weight")
plot(cumsum(weit)/(1:length(weit)),type="l", xlab="simulations", ylab="marginal likelihood")
boot=matrix(0,ncol=length(weit),nrow=100)
for (t in 1:100)
  boot[t,]=cumsum(sample(weit))/(1:length(weit))
uppa=apply(boot,2,quantile,.95)
lowa=apply(boot,2,quantile,.05)
polygon(c(1:length(weit),length(weit):1),c(uppa,rev(lowa)),col="gold")
lines(cumsum(weit)/(1:length(weit)),lwd=2)
plot(cumsum(weit)^2/cumsum(weit^2),type="l",xlab="simulations", ylab="Effective sample size",lwd=2)


x=rnorm(10^6)
wein=dcauchy(x)/dnorm(x)
#boxplot(wein/sum(wein))
plot(cumsum(wein*(x>2)*(x<6))/cumsum(wein),type="l")
abline(a=pcauchy(6)-pcauchy(2),b=0,col="sienna")

h <- function(x) 1.0/(x-1)
sam1=rt(.95*10^4,df=2)
sam2=1+.5*rt(.05*10^4,df=2)^2
sam=sample(c(sam1,sam2),.95*10^4)
weit=dt(sam,df=2)/(0.95*dt(sam,df=2) + .05*(sam>0)*dt(sqrt(2*abs(sam-1)),df=2)*sqrt(2)/sqrt(abs(sam-1)))
plot(cumsum(h(sam1))/(1:length(sam1)),ty="l")
lines(cumsum(weit*h(sam))/1:length(sam1),col="blue")

library(MASS)
glm(type~bmi,data=Pima.tr,family=binomial(link="probit"))
prior <- function(x) dnorm(x, 0, 100)
like<-function(beda){
  mia=mean(Pima.tr$bmi)
  prod(pnorm(beda[1]+(Pima.tr$bm[Pima.tr$t=="Yes"]- mia)*beda[2]))*
    prod(pnorm(-beda[1]-(Pima.tr$bm[Pima.tr$t=="No"]-mia)*beda[2]))/exp(sum(beda^2)/200)
}

post <- function (x){
  prior(x)*like(x)
}
  
sim=cbind(rnorm(10^3,mean=-.4,sd=.04), rnorm(10^3,mean=.065,sd=.005))
weit=apply(sim,1,post)/(dnorm(sim[,1],mean=-.4,sd=.04)*dnorm(sim[,2],mean=.065,sd=.005))
boxplot(weit)

sim=rbind(sim[1:(.95*10^3),],cbind(rnorm(.05*10^3,sd=10), rnorm(.05*10^3,sd=10)))
weit=apply(sim,1,post)/(.95*dnorm(sim[,1],m=-.4,sd=.081)*
                            dnorm(sim[,2],m=0.065,sd=.01)+.05*dnorm(sim[,1],sd=10)*
                            dnorm(sim[,2],sd=10))