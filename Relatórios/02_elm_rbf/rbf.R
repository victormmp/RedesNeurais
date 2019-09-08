rm(list=ls())
library(pracma)
library(corpcor)
library(mlbench)
library(stats)
load("data2classXOR.txt")

radial_activation <- function (X, centers, beta=0.1) {
    n_samples <- dim(X)[1]
    features <- dim(X)[2]

    n_centers <- dim(centers)[2]

    result <- matrix(nrow=n_samples, ncol=n_centers)

    for (sample in seq(n_samples)) {
        for (center in seq(n_centers)) {
            result[sample, center] <- exp(
                -beta * dist(rbind(X[sample,], centers[,center]))**2
            )
        }
    }

    return(result)
}

rows <- dim(X)[1]
features <- dim(X)[2]

# Number of neurons in hidden layer
p <- 5
Z <- replicate(p, replicate(features+1, 1))
print(Z)

# Training
Xaug <- cbind(replicate(rows, 1), X)

centroids <- kmeans(Xaug, p)
centers <- t(centroids$centers)

H <- radial_activation(Xaug, centers)

# H <- tanh(Xaug %*% Z)
W <- pseudoinverse(H) %*% Y

# Calculate Error
Y_hat <- sign(H %*% W)
err <- sum((Y-Y_hat)^2)/4
print(err)

# Test
Xaug <- cbind(replicate(rows, 1), X_t)
centers <- t(centroids$centers)
Ht <- radial_activation(Xaug, centers)
Y_hat_t <- sign(Ht %*% W)
err_t <- sum((Y_t-Y_hat_t)^2)/4
print(err_t)



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
p <- 5

# Training
Xaug <- cbind(replicate(rows, 1), x_train)
centroids <- kmeans(Xaug, p)
centers <- t(centroids$centers)
H <- radial_activation(Xaug, centers)
W <- pseudoinverse(H) %*% y_train

# Calculate Error
Y_hat <- sign(H %*% W)
err <- sum((y_train-Y_hat)^2)/4
print(err)


# Test
rows <- dim(x_test)[1]
features <- dim(x_test)[2]
Xaug <- cbind(replicate(rows, 1), x_test)
centers <- t(centroids$centers)
Ht <- radial_activation(Xaug, centers)
Y_hat_t <- sign(Ht %*% W)
err_t <- sum((y_test-Y_hat_t)^2)/4
print(err_t)
