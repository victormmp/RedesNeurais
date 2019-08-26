rm(list=ls())
library('corpcor')
library('pracma')
# load("data2classXOR.txt")

ssd <- 0.2
N <- 100
xc11 <- matrix(rnorm(N*2,mean=2,sd=0.4),ncol=2)
xc12 <- matrix(rnorm(N*2,mean=4,sd=0.4),ncol=2)

xc21 <- matrix(rnorm(N*2,mean=2,sd=0.4),ncol=2) + t(replicate(N, c(0,2)))
xc22 <- matrix(rnorm(N*2,mean=4,sd=0.4),ncol=2) + t(replicate(N, c(0,-2)))

plot(xc11, xlim=c(0,5), ylim = c(0,5),xlab = '', ylab = '', col='red')
par(new=T)
plot(xc12, xlim=c(0,5), ylim = c(0,5),xlab = '', ylab = '', col='red')
par(new=T)
plot(xc21, xlim=c(0,5), ylim = c(0,5),xlab = '', ylab = '', col='blue')
par(new=T)
plot(xc22, xlim=c(0,5), ylim = c(0,5),xlab = '', ylab = '', col='blue')

X1 <- rbind(xc11, xc12)
X2 <- rbind(xc21, xc22)
X <- rbind(X1, X2)

Y <- c(replicate(dim(X1)[1], -1), replicate(dim(X2)[1], 1))

rows <- dim(X)[1]
features <- dim(X)[2]

# Number of neurons in hidden layer
p <- 5
Z <- replicate(p, runif(features+1, -0.5, 0.5))
print(Z)
Xaug <- cbind(replicate(features * 4, 1), X)

H <- tanh(Xaug %*% Z)
W <- pseudoinverse(H) %*% Y

# Calculate Error
Y_hat <- sign(H %*% W)
err_t <- sum((Y-Y_hat)^2)/4
print(err_t)
# 
# line_y <- c()
# line_x <- seq(0, p, 0.01)
# for (xi in line_x){
#   yy <- c()
#   for (e in seq(p, 1)){
#     yy <- c(yy, xi ** e)
#   }
#   line_y <- c(line_y, dot(yy, W)) 
# }
# 
# par(new=T)
# plot(line_x, line_y, xlim=c(0,5), ylim = c(0,5))
