library(copula)
library(ggplot2)
library(ppcor)
library(VineCopula)

source('nonparanormal.R')

sampleSize <- 5000

D <- 6
simdata <- matrix(0, ncol=D, nrow=sampleSize)
simdata[, 1] <- rnorm(sampleSize, 0, 1)
rho <- 0.5
for (i in 2:D) {
  simdata[, i] <- rnorm(sampleSize, rho*simdata[, (i-1)], sqrt(1-rho^2))
}
# simdataQuantiles <- pnorm(simdata)

# mvgFit <- fitMVGaussianCopula(simdataQuantiles)
# Testing non-paranormal approx
topoOrder=1:D
corMatrixMN=cor(simdata)[1:D-1, 1:D-1]
vineCorParams=c(0.5)
fullCorMatrixMN <- computeFullCorMatrix(topoOrder, corMatrixMN, vineCorParams)
partialCors <- computePartialCorrelations(fullCorMatrixMN, topoOrder)
empCors <- calculateSequentialPartialCorrelations(cor(simdata))
all(abs(partialCors - empCors) <= 0.05)