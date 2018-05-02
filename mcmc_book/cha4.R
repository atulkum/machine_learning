h=function(x){(cos(50*x)+sin(20*x))^2}


x=matrix(h(runif(200*10^4)),ncol=200)
estint=apply(x,2,cumsum)/(1:10^4)
plot(estint[,1],ty="l",col=0,ylim=c(.8,1.2))
y=apply(estint,1,quantile,c(.025,.975))
#polygon(c(1:10^4,10^4:1),c(y[1,],rev(y[2,])),col="wheat")

boot=matrix(sample(x[,1],200*10^4,rep=T),nrow=10^4,ncol=200)
bootit=apply(boot,2,cumsum)/(1:10^4)
bootup=apply(bootit,1,quantile,.975)
bootdo=apply(bootit,1,quantile,.025)

polygon(c(1:10^4,10^4:1),c(bootup,rev(bootdo)),col="wheat")


norma=matrix(rnorm(500*10^4),ncol=500)+2.5
weit=1/(1+norma^2)
esti=apply(norma*weit,2,cumsum)/apply(weit,2,cumsum)
plot(esti[,1],type="l",col="white",ylim=c(1.7,1.9))
band=apply(esti,1,quantile,c(.025,.975))
polygon(c(1:10^4,10^4:1),c(band[1,],rev(band[2,])))

vare=cumsum(weit[,1]*norma[,1]^2)/cumsum(weit[,1])-esti[,1]^2
lines(esti[,1]+2*sqrt(vare/(1:10^4)))
lines(esti[,1]-2*sqrt(vare/(1:10^4)))

varw=cumsum(weit[,1]^2)*(1:10^4)/cumsum(weit[,1])^2
lines(esti[,1]+2*sqrt(varw*vare/(1:10^4)),col="sienna")
lines(esti[,1]-2*sqrt(varw*vare/(1:10^4)),col="sienna")

cocha=matrix(rcauchy(500*10^4),ncol=500)
range(cocha)
wach=dnorm(cocha,mean=2.5)
range(wach)

wachd=wach
wachd[apply(wachd,2,cumsum)<10^(-10)]=10^(-10)
range(wachd)

ess=apply(weit,2,cumsum)^2/apply(weit^2,2,cumsum)
essbo=apply(ess,1,quantile,c(.025,.975))
ech=apply(wachd,2,cumsum)^2/apply(wachd^2,2,cumsum)
echbo=apply(ech,1,quantile,c(.025,.975))

sumweit=apply(weit,2,cumsum)
plex=(apply(weit*log(weit),2,cumsum)/sumweit)-log(sumweit)
chumweit=apply(wachd,2,cumsum)
plech=(apply(wachd*log(wachd),2,cumsum)/chumweit)-log(chumweit)
plob=apply(exp(plex),1,quantile,c(.025,.975))
ploch=apply(exp(plech),1,quantile,c(.025,.975))


Nsim=10^4
norma=rnorm(Nsim)+2.5
hnorm=norma*dcauchy(norma)
munorm=mean(hnorm)
sdnorm=sd(hnorm)
f=function(x) (cumsum(hnorm))[round(Nsim*x)]/round(x*Nsim)
curve(munorm+(.1+3.15*sqrt(x))*sdnorm*10^2/round(x*Nsim), lwd=2,from=0,to=1)
curve(munorm-((.1+3.15*sqrt(x))*sdnorm*10^2/round(x*Nsim)), lwd=2,from=0,to=1,add=T)
curve(f,lwd=2,from=0.001,to=1,col="steelblue",add=T)

norma=rcauchy(Nsim)
hnorm=norma*dnorm(norma-2.5)
munorm=mean(hnorm)
sdnorm=sd(hnorm)
f=function(x) (cumsum(hnorm))[round(Nsim*x)]/round(x*Nsim)
curve(munorm+(.1+3.15*sqrt(x))*sdnorm*10^2/round(x*Nsim), lwd=2,from=0,to=1)
curve(munorm-((.1+3.15*sqrt(x))*sdnorm*10^2/round(x*Nsim)), lwd=2,from=0,to=1,add=T)
curve(f,lwd=2,from=0.001,to=1,col="steelblue",add=T)

Nsim=10^4
y=sqrt(rchisq(Nsim,df=nu)/nu)
x=rnorm(Nsim,mu,sigma/y)
d1=cumsum(exp(-x^2))/(1:Nsim)
d2=cumsum(exp(-mu^2/(1+2*(sigma/y)^2))/sqrt(1+2*(sigma/y)^2))/(1:Nsim)

nor=matrix(rnorm(Nsim*p),nrow=p)
risk=matrix(0,ncol=150,nrow=10)
a=seq(1,2*(p-2),le=10)
the=sqrt(seq(0,4*p,le=150)/p)
for (j in 1:150){
  nornor=apply((nor+rep(the[j],p))^2,2,sum)
  for (i in 1:10){
    for (t in 1:Nsim) 
      risk[i,j]=risk[i,j]+sum((rep(the[j],p)-max(1-a[i]/nornor[t],0)*(nor[,t]+rep(the[j],p)))^2)}}
risk=risk/Nsim

uref=runif(10^4)
x=h(uref)
estx=cumsum(x)/(1:10^4)
resid=uref%%2^(-q)
simx=matrix(resid,ncol=2^q,nrow=10^4)
simx[,2^(q-1)+1:2^1]=2^(-q)-simx[,2^(q-1)+1:2^1]
for (i in 1:2^q) simx[,i]=simx[,i]+(i-1)*2^(-q)
xsym=h(simx)
estint=cumsum(apply(xsym,1,mean))/(1:10^4)

thet=rnorm(10^3,mean=x)
delt=thet/(1+thet^2)
moms=delta=c()
for (i in 1:5){ 
  moms=rbind(moms,(thet-x)^(2*i-1)) 
  reg=lm(delt∼t(moms)-1)$coef 
  delta=rbind(delta,as.vector(delt-reg%*%moms))
}
plot(cumsum(delt)/(1:10^3),ty="l",lwd=2,lty=2)
for (i in 1:5) lines(cumsum(delta[i,])/(1:10^3),lwd=2)

glm(Pima.tr$t∼bmi,family=binomial)
sim=cbind(rnorm(10^4,m=-.72,sd=.55),rnorm(10^4,m=.1,sd=.2))
weit=apply(sim,1,like)/(dnorm(sim[,1],m=-.72,sd=.55)*dnorm(sim[,2],m=.1,sd=.2))
vari1=(1/(1+exp(-sim[,1]-sim[,2]*bmi)))-sum((Pima.tr$t=="Yes"))/length(Pima.tr$bmi)
vari2=(bmi/(1+exp(-sim[,1]-sim[,2]*bmi)))-sum(bmi[Pima.tr$t=="Yes"])/length(Pima.tr$bmi) 
resim=sample(1:Nsim,Nsim,rep=T,pro=weit)
reg=as.vector(lm(sim[resim,1]∼t(rbind(vari1[resim], + vari2[resim]))-1)$coef)