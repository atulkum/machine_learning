######
bm <- cumsum(rnorm(1000,0,1))
bm <- bm - bm[1]
plot(bm, main = "Brownian Motion", col = "blue", type = "l")
acf(diff(bm), main = "Autocorrelation of Wt")
par(mfrow = c(2,1))
hist(diff(bm), col = "orange", breaks = 100, main = "Wt-s Distribution")
qqnorm(diff(bm))
qqline(diff(bm))

library(quantmod)
getSymbols("AAPL")
price_AAPL <- AAPL[,6]
plot(price_AAPL, main = "The price of AAPL")

returns_AAPL <- diff(log(price_AAPL))
plot(returns_AAPL, main = "AAPL % returns", col = "blue")
hist(returns_AAPL, breaks = 100, col="brown")
acf(returns_AAPL[-1], main = "Autocorrelation plot of returns")
mR  <- mean(returns_AAPL[-1])
sdR <- sd(returns_AAPL[-1])

N     <- 1000
mu    <- 0.0010
sigma <- 0.025
p  <- c(100, rep(NA, N-1))
for(i in 2:N)
  p[i] <- p[i-1] * exp(rnorm(1, mu, sigma))
plot(p, type = "l", col = "brown", main = "Simulated Stock Price")

require(MASS)
require(quantmod)

#load a few symbols into memory
#"QQQQ",
getSymbols(c("AAPL",  "SPY", "GOOG", "CVX"))

#plot the prices of these stocks 
par(mfrow = c(3,2))
plot(AAPL[,6], main = "AAPL")
#plot(QQQQ[,6], main = "QQQQ")
plot(SPY[,6], main = "SPY")
plot(GOOG[,6], main = "GOOG")
plot(CVX[,6], main = "CVX")
par(mfrow = c(1,1))

#compute price matrix
#QQQQ[,6],
pM <- cbind(AAPL[,6],  SPY[,6], GOOG[,6], CVX[,6])

#compute returns matrix
rM <-  apply(pM,2,function(x) diff(log(x)))

#look at pairwise charts
pairs(coredata(rM))

#compute the covariance matrix
covR <- cov(rM)

#use this covariance matrix to simulate normal random numbers
#that share a similar correlation structure with the actual data
meanV <- apply(rM, 2, mean)
rV    <- mvrnorm(n = nrow(rM), mu = meanV, Sigma = covR)

#simulate prices based on these correlated random variables

#calculate mean price
p0 <- apply(pM,2,mean)
sPL <- list()
for(i in 1:ncol(rM)){
  sPL[[i]] <-round(p0[i]*exp(cumsum(rV[,i])),2)
}

#plot simulated prices
par(mfrow = c(3,2)) 
plot(sPL[[1]],main="AAPLsim",type="l")
#plot(sPL[[2]], main = "QQQQ sim",type = "l")
plot(sPL[[2]], main = "SPY sim", type = "l") 
plot(sPL[[3]], main = "GOOG sim",type = "l") 
plot(sPL[[4]], main = "CVX sim", type = "l")
###