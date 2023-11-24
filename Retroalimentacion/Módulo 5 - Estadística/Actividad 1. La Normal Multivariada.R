library(mnormt)

mu <- c(2.5, 4)
sigma <- matrix(c(1.2, 0, 0, 2.3), nrow=2)
x <- c(2, 3)

prob <- pmnorm(x, mu, sigma)
print(prob)

library(mnormt)

x_seq <- seq(-1, 6, 0.1)
y_seq <- seq(0, 8, 0.1)
mu <- c(2.5, 4)
sigma <- matrix(c(1.2, 0, 0, 2.3), nrow=2)

f <- function(x, y) dmnorm(cbind(x, y), mu, sigma)
z <- outer(x_seq, y_seq, f)

persp(x_seq, y_seq, z, theta=-30, phi=25, expand=0.6, ticktype='detailed', col="pink")

library(mnormt)

x_seq <- seq(-1, 6, 0.1)
y_seq <- seq(0, 8, 0.1)
mu <- c(2.5, 4)
sigma <- matrix(c(1.2, 0, 0, 2.3), nrow=2)

f <- function(x, y) dmnorm(cbind(x, y), mu, sigma)
z <- outer(x_seq, y_seq, f)

contour(x_seq, y_seq, z, col="blue", levels=c(0.01, 0.03, 0.05, 0.07, 0.09))


