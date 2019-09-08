library('tictoc')
library('stringr')
library('pracma')
library('mlbench')

# Criação dos dados

# set.seed(42)
# f_generator <- c(2, 1)
# y <- c()
# x <- seq(0, 12, 0.05)
# for (index in x) {
#     data_y <- index * f_generator[1] + f_generator[2]
#     y <- c(y, data_y + rnorm(1, 0, 1))
# }
#
# x <- as.matrix(x, ncol=1)
# y <- as.matrix(y, ncol=1)
#
# index_train = sample(dim(x)[1], replace=FALSE)
# x_train = x[index_train,]
# y_train = y[index_train,]
#
# xlim <- c(0,12)
# ylim <- c(0, 30)
# plot(x, y, xlab = 'x1', ylab = 'x2', xlim=xlim, ylim=ylim)

# Definição das funções de treinamento e teste

train <- function(x, y, eta=0.01, dimension=2) {

    print("Starting training function...")
    tic("Training complete.")

    epochs <- 100
    correction_factor <- eta

    x_aug <- cbind(x, replicate(dim(x)[1], 1))

    w = rnorm(dim(x)[2]+1, 0, 1)

    errors = c()

    for (iter in seq(epochs)) {
        print(str_interp('Epoch [ ${iter} / ${epochs} ]'))
        x_seq <- sample(dim(x_aug)[1])

        curr_error <- 0
        for (index in x_seq){
            y_partial <- 1*(x_aug[index,] %*% w)
            error <- y[index,1] - y_partial
            delt_w <- correction_factor*error*x_aug[index,]
            w <- w + delt_w

            curr_error <- curr_error + error**2
        }
        curr_error <- curr_error / dim(x)[1]
        print(str_interp('    Error: ${curr_error}'))
        errors <- c(errors, curr_error)
    }

    toc()
    result <- list('errors'=errors, 'weights'=w)

    return(result)
}

fit <- function(x, w) {
    x_aug <- cbind(x, replicate(dim(x)[1], 1))

    rows <- dim(x_aug)[1]
    cols <- dim(x_aug)[2]

    y <- matrix(ncol = 1, nrow = rows)

    for (index in seq(rows)) {
        y[index,] <- x_aug[index, ] %*% w
    }

    return (y)

}

evaluate <- function(y, y_hat) {
    rows <- dim(y)

    mse <- 0
    err_percent <- 0
    for (index in seq(rows)) {
        error <- y[index,] - y_hat[index,]
        err_percent <- error / y[index,]
        mse <- mse + error**2
    }

    return (list(
        'mse'=mean(mse),
        'err_percent'=mean(err_percent)
    ))
}

# # Resultados
# results <- train(x, y)
# # plot(results$errors, xlab = 'Iteração', ylab = 'MSE')
#
# line_y <- c()
# line_x <- x
# w <- results$weights
# for (xi in line_x){
#     x2_val <- w[2] + w[1] * xi
#     line_y <- c(line_y, x2_val)
# }
# par(new=T)
# plot(line_x, line_y, xlab='', ylab='', xlim=xlim, ylim = ylim, type='l', col='red')
#


##############################################################

rm(list = setdiff(ls(), lsf.str()))
data = read.csv("BUILDING1paraR.DT", sep=" ")

x <- as.matrix(data[1:14])
y <- as.matrix(data[15:17])

y_energy <- as.matrix(data[15])
y_hot <- as.matrix(data[16])
y_cold <- as.matrix(data[17])


index_train = sample(seq(dim(x)[1]), as.integer(0.7*dim(x)[1]), replace=FALSE)
x_train = as.matrix(x[index_train,])
x_test = as.matrix(x[-index_train,])

y_train <- as.matrix(y[index_train,])
y_test = as.matrix(y[-index_train,])


results_energy <- train(x_train, as.matrix(y_train[,1]))
results_hot <- train(x_train, as.matrix(y_train[,2]))
results_cold <- train(x_train, as.matrix(y_train[,3]))

y_energy <- fit(x_test, results_energy$weights)
y_hot <- fit(x_test, results_hot$weights)
y_cold <- fit(x_test, results_cold$weights)

mse_energy <- evaluate(as.matrix(y_test[, 1]), y_energy)
mse_hot <- evaluate(as.matrix(y_test[, 2]), y_hot)
mse_cold <- evaluate(as.matrix(y_test[, 3]), y_cold)
