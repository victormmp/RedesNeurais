rm(list=ls())
library(tictoc)

print('Inicialização dos pesos')

w95 <-runif(1) -0.5
w96 <-runif(1) -0.5
w97 <-runif(1) -0.5

w106 <-runif(1) -0.5
w107 <-runif(1) -0.5
w108 <-runif(1) -0.5

w61 <-runif(1) -0.5
w62 <-runif(1) -0.5
w63 <-runif(1) -0.5

w72 <-runif(1) -0.5
w73 <-runif(1) -0.5
w74 <-runif(1) -0.5

print('Inicialização dos bias')

i1 <- 1
i4 <- 1
i5 <- 1
i8 <- 1

print('Inicialização das entradas')

xall <- matrix(c(0,0,0,1,1,0,1,1), ncol=2, byrow=T)
yall <- matrix(c(-1,1,1,-1,1,-1,-1,1), ncol=2, byrow=T)

print('Inicialização dos parâmetros de treinamento')

tol <- 0.1
epochs <- 100
nepocas <- 0
eepoca <- tol + 1
n_neurons <- 2

x_aug <- cbind(replicate(dim(xall)[1], 1), xall)

rows <- dim(x_aug)[1]
cols <- dim(x_aug)[2]
n_out <- dim(yall)[2]

print('Loop de treinamento')

eta <- 0.01
# Hidden layer
z <- matrix(rnorm(cols * n_neurons, 0, 1), ncol=n_neurons)

# Out layer
w <- matrix(rnorm(cols * n_neurons, 0, 1), ncol=n_neurons)

errors = c()

error <- tol + 1
epoch <- 0
while ((epoch < epochs)) {
    print(paste('Epoch [', epoch, '/', epochs, ']'))

    for (index in seq(rows)) {

        # Feed foward
        x <- x_aug[index,]
        y1 <- tanh(x %*% w)
        y1_aug <- cbind(1, y1)
        y2 <- tanh(y1_aug %*% z)
        error <- yall[index,] - y2

        # Backpropagation
        z <- z + eta * x %*% error

    }

    epoch <- epoch + 1
}

result <- list('errors'=errors, 'weights'=w)


