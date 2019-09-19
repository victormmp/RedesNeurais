rm(list=ls())
library(ggplot2)
library(tictoc)
library(stringr)

set.seed(42)
class1 = matrix(rnorm(800, 3, 2), ncol = 2)
class2 = matrix(rnorm(800, 7, 2), ncol = 2)

lim <- c(-3,12)
plot(class1[,1], class1[,2], xlim=lim, ylim=lim, col='red', xlab='x1', ylab='x2')
par(new=TRUE)
plot(class2[,1], class2[,2], xlim=lim, ylim=lim, col='blue', xlab='', ylab='')

x = rbind(class1, class2)
y = c(replicate(dim(class1)[1], 0), replicate(dim(class2)[1], 1))

index_train = sample(seq(dim(x)[1]), as.integer(0.7*dim(x)[1]), replace=FALSE)
x_train = x[index_train,]
x_test = x[-index_train,]

y_train <- y[index_train]
y_test = y[-index_train]

train <- function(x, y, eta=0.01) {

    print("Starting training function...")
    tic("Training complete.")

    epochs <- 100
    correction_factor <- eta

    x_aug <- cbind(x, replicate(dim(x)[1], 1))

    w = rnorm(dim(x)[2] +1, 0, 1)

    errors = c()

    for (iter in seq(epochs)) {
        print(str_interp('Epoch [ ${iter} / ${epochs} ]'))

        curr_error <- c()
        for (index in sample(dim(x)[1])){
            y_partial <- 1*((x_aug[index,] %*% w) >= 0)
            error <- y[index] - y_partial
            delt_w <- correction_factor*error*x_aug[index,]
            w <- w + delt_w

            curr_error <- c(curr_error, error)
        }
        print(str_interp('    Error: ${mean(curr_error)}'))
        errors <- c(errors, mean(curr_error))
    }

    toc()
    result <- list('errors'=errors, 'weights'=w)

    return(result)
}


evaluate <- function(x, y, w) {
    y_result <- c()

    errors <- c()
    false_positive <- 0
    true_positive <- 0
    false_negative <- 0
    true_negative <- 0

    x_aug <- cbind(x, replicate(dim(x)[1], 1))

    for (index in seq(dim(x)[1])) {
        y_partial <- 1*((x_aug[index,] %*% w) >= 0)
        error <- y[index] - y_partial

        if (error > 0) {
            false_negative <- false_negative + 1
        } else if (error < 0) {
            false_positive <- false_positive + 1
        } else if (y[index] == 1) {
            true_positive <- true_positive + 1
        } else {
            true_negative <- true_negative + 1
        }

        errors <- c(errors, error)
        y_result <- c(y_result, y_partial)

    }

    confusion_matrix <- matrix(replicate(4, 0), nrow = 2, ncol = 2)
    confusion_matrix[1,1] <- true_positive
    confusion_matrix[1,2] <- false_positive
    confusion_matrix[2,1] <- false_negative
    confusion_matrix[2,2] <- true_negative

    return(list(
        'y_result' = y_result,
        'errors' = errors,
        'mean_error'= mean(errors),
        'accuracy' = (1 - mean(errors)),
        'specitivity'= true_negative / (true_negative + false_positive),
        'sensibility' = true_positive / (true_positive + false_negative),
        'confusion_matrix' = confusion_matrix
    ))
}

# Distancia dos pontos à reta ótima
xc1_aug <- cbind(-1, class1)
xc2_aug <- cbind(-1, class2)

distc1r <- 5

result <- train(x_train, y_train)
evaluate(x_test, y_test, result$weights)
