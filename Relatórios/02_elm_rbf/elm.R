rm(list=ls())
library(pracma)
library(corpcor)
library(mlbench)
load("data2classXOR.txt")

rows <- dim(X)[1]
features <- dim(X)[2]

# Number of neurons in hidden layer
p <- 5
Z <- replicate(p, runif(features+1, -0.5, 0.5))
print(Z)

# Training
Xaug <- cbind(replicate(rows, 1), X)

H <- tanh(Xaug %*% Z)
W <- pseudoinverse(H) %*% Y

# Calculate Error
Y_hat <- sign(H %*% W)
err <- sum((Y-Y_hat)^2)/4
print(err)

# Test
Xaug <- cbind(replicate(features * 4, 1), X_t)
Ht <- tanh(Xaug %*% Z)
Y_hat_t <- sign(Ht %*% W)
err_t <- sum((Y_t-Y_hat_t)^2)/4
print(err_t)



# x_seq <- seq(0, 6, 0.1)
# N <- length(x_seq)
# M <- matrix(ncol=length(x_seq), nrow=length(x_seq))
#
# for (i in seq(N)) {
#     for (j in seq(N)) {
#         M[i, j] <- x_seq
#     }
# }


data("BreastCancer")

bc <- BreastCancer[complete.cases(BreastCancer),]
x <- bc[, 2:10]
y <- bc[,11]

x <- sapply(x, as.numeric)
y <- replicate(dim(bc)[1], 0)
y[which(bc[,11]== 'benign')] = 0
y[which(bc[,11]== 'malignant')] = 1

index_train = sample(seq(dim(x)[1]), as.integer(0.7*dim(x)[1]), replace=FALSE)
x_train = x[index_train,]
x_test = x[-index_train,]

y_train <- y[index_train]
y_test = y[-index_train]

rows <- dim(x_train)[1]
features <- dim(x_train)[2]

# Number of neurons in hidden layer
p <- 20
Z <- replicate(p, runif(features+1, -0.5, 0.5))
Xaug <- cbind(replicate(rows, 1), x_train)

H <- tanh(Xaug %*% Z)
W <- pseudoinverse(H) %*% y_train

# Calculate Error
Y_hat <- sign(H %*% W)
err <- sum((y_train-Y_hat)^2)/4
print(err)


# Test
rows <- dim(x_test)[1]
features <- dim(x_test)[2]

# Number of neurons in hidden layer
p <- 20
Z <- replicate(p, runif(features+1, -0.5, 0.5))
Xaug <- cbind(replicate(rows, 1), x_test)

H <- tanh(Xaug %*% Z)
Y_hat <- sign(H %*% W)
err <- sum((y_test-Y_hat)^2)/4
print(err)
