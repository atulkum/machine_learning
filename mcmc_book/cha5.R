xm=rcauchy(500)
f=function(y){-sum(log(1+(x-y)^2))}
for (i in 1:500){
   x=xm[1:i]
   mi=optimise(f,interval=c(-10,10),maximum=T)$max
}

f=function(y){-sin(y*100)^2-sum(log(1+(x-y)^2))}
for (i in 1:500){
  x=xm[1:i]
  mi1=optimise(f,interval=c(-10,10),maximum=T)$max
}
da=rbind(rnorm(10^2),2.5+rnorm(3*10^2))
like=function(mu){sum(log((.25*dnorm(da-mu[1])+.75*dnorm(da-mu[2]))))}
sta=c(1,1)
mmu <- sta
itr <- nlm(like,sta)$it
for (i in 1:itr)
  mmu=rbind(mmu,nlm(like,sta,iterlim=i)$est)
plot()
plot(mmu,pch=19,lwd=2)

h <- function(x) {(cos(50*x) + sin(20*x))^2}
rangom=h(matrix(runif(10^6),ncol=10^3))
monitor=t(apply(rangom,1,cummax))
plot(monitor[1,],type="l",col="white")
polygon(c(1:10^3,10^3:1),c(apply(monitor,2,max),
                                    rev(apply(monitor,2,min))),col="grey")
abline(h=optimise(h,int=c(0,1),maximum=T)$ob)


cau=rcauchy(10^2)
mcau=median(cau)
rcau=diff(quantile(cau,c(.25,.75)))
f=function(x){
   z=dcauchy(outer(x,cau,FUN="-"))
   apply(z,1,mean)}
fcst=integrate(f,lower=-20,upper=20)$value
ft=function(x){f(x)/fcst}
g=function(x){dt((x-mcau)/rcau,df=49)/rcau}
curve(ft,from=-10,to=10)
curve(g,add=T)

dft = (101 -1)/2
unisan=matrix(f(runif(5*10^4,-5,5)),ncol=500)
causan=matrix(f(rt(5*10^4,df=dft)*rcau+mcau),ncol=500)
unimax=apply(unisan,2,cummax)[10:10^2,]
caumax=apply(causan,2,cummax)[10:10^2,]
plot(caumax[,1],col="white",ylim=c(.8,1)*max(causan))
polygon(c(10:10^2,10^2:10),c(apply(unimax,1,max),
                               rev(apply(unimax,1,min))),col="grey")
polygon(c(10:10^2,10^2:10),c(apply(caumax,1,max),
                               rev(apply(caumax,1,min))),col="wheat")

h=function(x,y){(x*sin(20*y)+y*sin(20*x))^2*cosh(sin(10*x)*x)+(x*cos(10*y)-y*sin(10*x))^2*cosh(cos(20*y)*y)}
x=y=seq(-3,3,le=435)             #defines a grid for persp
z=outer(x,y,h)
par(bg="wheat",mar=c(1,1,1,1))   #bg stands for background
persp(x,y,z,theta=155,phi=30,col="green4",  ltheta=-120,shade=.75,border=NA,box=FALSE)

h=function(x,y){(x*sin(20*y)+y*sin(20*x))^2*cosh(sin(10*x)*x)+(x*cos(10*y)-y*sin(10*x))^2*cosh(cos(20*y)*y)}
start=c(.65,.8)
theta=matrix(start,ncol=2)
diff=iter=1
while (diff>10^-5){
  zeta=rnorm(2)
  zeta=zeta/sqrt(t(zeta)%*%zeta)
  grad=alpha[iter]*zeta*(h(theta[iter,]+beta[iter]*zeta)-
                                      h(theta[iter,]-beta[iter]*zeta))/beta[iter]
  theta=rbind(theta,theta[iter,]+grad)
  dif=sqrt(t(grad)%*%grad)
  iter=iter+1
}
alpha <- function(j){
  1/log(j + 1)
}

beta <- function(j){
  alpha(j)^0.1
}


sdg1 <- function(){
  start=c(.65,.8)
  theta=matrix(start,ncol=2)
  diff=iter=1
  while (diff>10^-5){
    scale=sqrt(t(grad)%*%grad)
    alpha_i = alpha(iter)
    beat_i = beta(iter)
    grad = NULL
    while (scale>1){
      zeta=rnorm(2);
      zeta=zeta/sqrt(t(zeta)%*%zeta)
      grad=alpha_i*zeta*(h(theta[iter,]+beat_i*zeta)-
                             h(theta[iter,]-beat_i*zeta))/beat_i
      scale=sqrt(t(grad)%*%grad)
    }
    theta=rbind(theta,theta[iter,]+grad)
    dif=sqrt(t(grad)%*%grad)
    iter=iter+1
  }
}


sdg1()

theta=rep(theta0,Nsim)
hcur=h(theta0)
xis=randg(Nsim)
for (t in 2:Nsim){
  prop=theta[t-1]+xis[t]
  hprop=h(prop)
  if (Temp[t]*log(runif(1))<hprop-hcur){
    theta[t]=prop
    hcur=hprop
  }else{
    theta[t]=theta[t-1]
  }
}
h <- function(x) {(cos(50*x) + sin(20*x))^2}

x=runif(1)
hval=hcur=h(x)
diff=iter=1
while (diff>10^(-4)){
  temp_iter = 1/(1 + iter)^2 #1/log(1+iter)
  scale = 0.5*sqrt(temp_iter)#5*sqrt(temp_iter)#sqrt(temp_iter)
  prop=x[iter]+runif(1,-1,1)*scale
  if ((prop>1)||(prop<0)|| (log(runif(1))*temp_iter>h(prop)-hcur))
    prop=x[iter]
  x=c(x,prop)
  hcur=h(prop)
  hval=c(hval,hcur)
  if ((iter>10)&&(length(unique(x[(iter/2):iter]))>1))
        diff=max(hval)-max(hval[1:(iter/2)])
  iter=iter+1
}
hval
optimise(h,int=c(0,1),maximum=T)$obj


SA=function(x){
  temp=scale=iter=dif=factor=1
  the=matrix(x,ncol=2)
  curlike=hval=like(x)
  while (dif>10^(-4)){
    prop=the[iter,]+rnorm(2)*scale[iter]
    if ((max(-prop)>2)||(max(prop)>5)||
          (temp[iter]*log(runif(1))>-like(prop)+curlike))
      prop=the[iter,]
    curlike=like(prop);hval=c(hval,curlike);the=rbind(the,prop)
    iter=iter+1;temp=c(temp,1/10*log(iter+1))
    ace=length(unique(the[(iter/2):iter,1]))
    if (ace==1) factor=factor/10
    if (2*ace>iter) factor=factor*10
    scale=c(scale,max(2,factor*sqrt(temp[iter])))
    dif=(iter<100)+(ace<2)+(max(hval)-max(hval[1:(iter/2)]))
  }
  list(theta=the,like=hval,ite=iter)
}

prop=the[iter,]+scale[iter]*rnorm(2)
scale=min(.1,5*factor*sqrt(temp[iter]))
title(main=paste("min",format(-max(hval),dig=3),sep=" "))

margap=function(a){
  b=rt(10^3,df=5)
  dtb=dt(b,5,log=T)
  b=b*.1+.1
  themar=0
  themar/10^3
}
m=1 
i=1
themar=0
for (i in 1:10^3)
  themar=themar+exp(like(a,b[i])-dtb[i])

like=function(a,b){
  apply(pnorm(-a-outer(X=b,Y=da[,2],FUN="*"),lo=T)*(1-da[,1])
      pnorm(a+outer(X=b,Y=da[,2],FUN="*"),lo=T)*da[,1],1,sum)}


theta=rnorm(1,mean=ybar,sd=sd(y))
iteronstop=1
while (nonstop){
   theta=c(theta,m*ybar/n+(n-m)*(theta[iter]+
                                       dnorm(a-theta[iter])/pnorm(a-theta[iter]))/n)
   iter=iter+1
   nonstop=(diff(theta[iter:(iter+1)])>10^(-4)) }

mcmc(beta,sigma,x,y,T)
targ=function(beta,x,y,uni){
  xs=exp(beta*x)
  xxs=x*xs
  ome=exp(uni)
  prodct=0
  for (j in 1:m) for (t in 1:T)
    prodct=prodct+sum(xxs[,j]*ome[,t]/(1+xs[,j]*ome[,t]))
  prodct-sum(T*x*y)
}

beta=uniroot(targ,x=x,y=y,u=u,int=mlan+10*sigma*c(-1,1))
mlan=as.numeric(glm(as.vector(y)∼as.vector(x)-1, fa=binomial)$coe)
T=1000     #Number of MCEM simulations
beta=mlan
sigma=diff=iter=factor=1
while (diff>10^-3){
    samplu=mcmc(beta[iter],sigma[iter],x,y,T)
  sigma=c(sigma,sd(as.vector(samplu)))
  beta=c(beta,uniroot(targ,x=x,y=y,u=samplu,
                      inter=mlan+c(-10*sigma,10*sigma)))
  diff=max(abs(diff(beta[iter:(iter+1)])),
           abs(diff(sigma[iter:(iter+1)])))
  iter=iter+1
  T=T*2}
like=function(beta,sigma){
  lo=0
  for (t in 1:(10*T)){
    uu=rnorm(n)*sigma
    lo=lo+exp(sum(as.vector(y)*(beta*as.vector(x)+rep(uu,m)))-
                lo/T }

