library(coda)
T=10^3
beta=mlan
sigma=1
u=rnorm(n)*sigma
samplu=matrix(u,nrow=n)
for (iter in 2:T){
  u=rnorm(n)
  for (i in 1:n){
     mu=samplu[i,iter-1]
     u[i]=sigma[iter-1]*rnorm(1)+mu
      if (log(runif(1))>gu(u[i],i,beta[iter-1],sigma[iter-1])-gu(mu,i,beta[iter-1],sigma[iter-1]))
          u[i]=mu
  }
  samplu=cbind(samplu,u)
  sigma=c(sigma,1/sqrt(2*rgamma(1,0.5*n)/sum(u^2)))
  tau=sigma[iter-1]/sqrt(sum(as.vector(x^2)*pro(beta[iter-1],u)))
  betaprop=beta[iter-1]+rnorm(1)*tau
  if (log(runif(1))>likecomp(betaprop,sigma[iter],u)-likecomp(beta[iter-1],sigma[iter],u))
      betaprop=beta[iter-1]
  beta=c(beta,betaprop) 
}

mcmc(cbind(beta,sigma))
cumuplot
plot(mcmc(t(samplu)))
heidel.diag(mcmc(beta))