library(copula)
library(CondIndTests)
library(ggplot2)
library(gridExtra)
library(ppcor)
library(tidyverse)
library(VineCopula)

source('nonparanormal.R')

sampleSize <- 300
D <- 4

# Making Clayton model
structureMatrix <- matrix(
                   c(1, 4, 3, 2,
                     0, 2, 4, 3,
                     0, 0, 3, 4,
                     0, 0, 0, 4), ncol=D)
familyMatrix <- matrix(
  c(0, 0, 0, 4,
    0, 0, 0, 4,
    0, 0, 0, 1,
    0, 0, 0, 0), ncol=D)

clayton_dep <- +5
normal_corr <- 0.99
parameterMatrix <- matrix(
                   c(0, 0, 0, clayton_dep,
                     0, 0, 0, clayton_dep,
                     0, 0, 0, normal_corr,
                     0, 0, 0, 0), ncol=D)


normal_corr_values <- seq(0.38, 0.98, by = 0.2)
# normal_corr_values <- c(0.3, 0.8)
set.seed(1)
simulateAndPlot(
  structureMatrix, 
  familyMatrix,
  sampleSize=1000, 
  1:D, 
  normal_corr_values, 
  general_dep=clayton_dep, 
  general_family=4, 
  seed=NULL
)

# Sample from Vine
vineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, sampleSize)
# gaussCop <- normalCopula(dim = D, dispstr = "un")
# 
# # Fit the MVG copula model to the data using MLE
# fit <- fitCopula(gaussCop, data = vineOutput$simdata, method = 'itau')
# 
# vineCopFit <- RVineStructureSelect(data = vineOutput$simdata, familyset = 1)
# 
# mvgFit <- fitMVGaussianCopula(vineOutput$simdata, method='itau')
# # contour(vineOutput$RVM)
# 
# # Settings for reparameterisation
# topoOrder=1:D
# corMatrixMN=mvgFit$correlationMatrix[1:(D-1), 1:(D-1)]
# vineCorParams=normal_corr
# 
# npReparamVine <- simulateAndReparameterizeVine(
#   structureMatrix, 
#   familyMatrix, 
#   parameterMatrix, 
#   sampleSize, 
#   topoOrder, 
#   vineCorParams
# )
# 
# ###########################
# ############ HACKY SOLUTION
# ###########################
newVineOutput <- npReparamVine$newVineOutput

# Final variable is the outcome
newSimData <- newVineOutput$simdata
covariate_ranks <- newSimData[, 1:(dim(newSimData)[2] - 1)]
outcome_model <-multivariate_conditional_mean_and_samples(
  X2_samples = qnorm(covariate_ranks),
  R = npReparamVine$fullCorMatrixMN
)
outcome_quantile_samples <- pnorm(outcome_model$generated_samples)
newSimData[, dim(newSimData)[2]] <- outcome_quantile_samples
npReparamVine$newVineOutput$simdata <- newSimData
newVineOutput$simdata <- newSimData
# ###########################
# ###########################
# ###########################
# 
# oldVineOutput <- npReparamVine$oldVineOutput
# print(cor(qnorm(npReparamVine$oldVineOutput$simdata)))
# print(cor(qnorm(newSimData)))
# # contour(newVineOutput$RVM)
# margins <- oldVineOutput$simdata
# margins_np <- newVineOutput$simdata
# 
# ggplot(tibble(v1=qnorm(oldVineOutput$simdata[, 2]), v6=qnorm(oldVineOutput$simdata[, 4])), aes(x=v1, y=v6)) + 
#   # geom_point(alpha=0.2) + 
#   geom_density2d_filled(alpha=0.6) +
#   labs(title='True Marginal Dependence between V4 and V6') +
#   xlim(-3, 3) +
#   ylim(-3, 3)
# 
# ggplot(tibble(v1=qnorm(newVineOutput$simdata[, 2]), v6=qnorm(newVineOutput$simdata[, 4])), aes(x=v1, y=v6)) + 
#   # geom_point(alpha=0.2) + 
#   geom_density2d_filled(alpha=0.6) +
#   labs(title='NP Marginal Dependence between V4 and V6') +
#   xlim(-3, 3) +
#   ylim(-3, 3)
#  # We check that the partial correlations are indeed zero for V6 and the other variables.
# 
# # ggpairs(qnorm(margins_np))
# # We know what the 56 margin should be (normal gaussian)
# # Can compare that to the marginal of 45 (which should be clayton)
# # Check that this is uniform for the true distribution and check if true for non-paranormal approx
# F4_5_np <- BiCopHfunc(margins_np[,2], margins_np[,3], family=4, par=clayton_dep)$hfunc2
# emp_corr <- cor(qnorm(margins_np[,4]), qnorm(margins_np[,3]))
# F6_5_np <- BiCopHfunc(margins_np[,4], margins_np[,3], family=1, par=emp_corr)$hfunc2
# cor.test(F4_5_np, F6_5_np, method=c("kendall"))
# 
# ggplot(data.frame(F5 = qnorm(margins_np[,3]), F6 = qnorm(margins_np[,4])), aes(x = F5, y = F6)) +
#   geom_density2d_filled(alpha=0.6) +
#   labs(x = "F5", y = "F6", title = "NP Contour Plot of F5 vs F6") +
#   theme_minimal()
# ggplot(data.frame(F5 = qnorm(margins[,3]), F6 = qnorm(margins[,4])), aes(x = F5, y = F6)) +
#   geom_density2d_filled(alpha=0.6) +
#   labs(x = "F5", y = "F6", title = "True Contour Plot of F5 vs F6") +
#   theme_minimal()
# 
# 
# 
# # Generate example data
# set.seed(123)
# X <- qnorm(margins[, 2])
# Y <- qnorm(margins[, 4])
# Z <- qnorm(margins[, 3])
# 
# # Perform Kernel Conditional Independence Test
# kci_result <- KCI(X, Y, Z)
# print(kci_result)
# 
# X_np <- qnorm(margins_np[, 2])
# Y_np <- qnorm(margins_np[, 4])
# Z_np <- qnorm(margins_np[, 3])
# 
# # Perform Kernel Conditional Independence Test
# kci_result_np <- KCI(X_np, Y_np, Z_np)
# print(kci_result_np)
# 
# 
# F4_3 <- BiCopHfunc(margins[,4], margins[,3], family=1, par=normal_corr)$hfunc2
# F2_3 <- BiCopHfunc(margins[,2], margins[,3], family=4, par=clayton_dep)$hfunc2
# cor.test(F4_3, F2_3, method=c("kendall"))
# 
# 
# p1 <- ggplot(data.frame(F4_5 = qnorm(F4_5_np), F6_5 = qnorm(F6_5_np)), aes(x = F4_5, y = F6_5)) +
#   geom_density2d_filled(alpha=0.6) +
#   labs(x = "F4_5", y = "F6_5", title = "NP Contour Plot of F4_5 vs F6_5") +
#   theme_minimal() + 
#   xlim(-3, 3) +
#   ylim(-3, 3)
# 
# # Plot for the true distribution
# F4_3 <- BiCopHfunc(margins[,4], margins[,3], family=1, par=normal_corr)$hfunc2
# F2_3 <- BiCopHfunc(margins[,2], margins[,3], family=4, par=clayton_dep)$hfunc2
# p2 <- ggplot(data.frame(F4_3 = qnorm(F4_3), F2_3 = qnorm(F2_3)), aes(x = F4_3, y = F2_3)) +
#   geom_density2d_filled(alpha=0.6) +
#   labs(x = "F4_3", y = "F2_3", title = "True Contour Plot of F4_3 vs F2_3") +
#   theme_minimal() + 
#   xlim(-3, 3) +
#   ylim(-3, 3)
# 
# # Combine the plots side by side
# grid.arrange(p1, p2, ncol = 2)
# 
# x <- pnorm(rnorm(sampleSize))
# y <- pnorm(rnorm(sampleSize, rho*x, sqrt(1-rho^2)))
# z <- pnorm(rnorm(sampleSize, rho*y, sqrt(1-rho^2)))
# d <- data.frame(x=x,y=y,z=z)
# Fx_y <- BiCopHfunc(x, y, family=1, par=rho)$hfunc2
# Fz_y <- BiCopHfunc(z, y, family=1, par=rho)$hfunc2
# ggplot(data.frame(Fz_y=Fz_y, Fx_y=Fx_y), aes(x = Fz_y, y = Fx_y)) +
#   geom_density_2d() + 
#   labs(x = "Fz_y", y = "Fx_y", title = "Contour Plot of Fz_y vs Fx_y") +
#   theme_minimal()
