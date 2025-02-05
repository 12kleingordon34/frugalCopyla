library(copula)
library(CondIndTests)
library(ggplot2)
library(gridExtra)
library(ppcor)
library(tidyverse)
library(VineCopula)

source('nonparanormal.R')

sampleSize <- 300
D <- 3

# Making Clayton model
structureMatrix <- matrix(
                   c(1, 3, 2,
                     0, 2, 3,
                     0, 0, 3), ncol=D)
familyMatrix <- matrix(
  c(0, 0, 4,
    0, 0, 1,
    0, 0, 0), ncol=D)

clayton_dep <- +3
rho_12 <-  0.848# (clayton_dep - 1) / clayton_dep
rho_23 <- 0.98
parameterMatrix <- matrix(
                   c(0, 0, clayton_dep,
                     0, 0, rho_23,
                     0, 0, 0), ncol=D)
# 
# cor_matrix <- matrix(
#   c(1, rho_12, rho_13,
#     rho_12, 1, rho_23,
#     rho_13, rho_23, 1), ncol=D)
# normal_corr <- rho_23
# rho_13 = rho_12 * rho_23
# rho_23_1 = (rho_23 - rho_12 * rho_13) / (sqrt(1 - rho_12^2) * sqrt(1 - rho_13^2))
# 
# # Making Clayton model
# structureMatrix <- matrix(
#   c(2, 3, 1,
#     0, 1, 3,
#     0, 0, 3), ncol=D)
# familyMatrix <- matrix(
#   c(0, 1, 1,
#     0, 0, 1,
#     0, 0, 0), ncol=D)
# parameterMatrix <- matrix(
#   c(0, rho_23_1, rho_12,
#     0, 0, rho_13,
#     0, 0, 0), ncol=D)


normal_corr_values <- seq(0.38, 0.98, by = 0.2)
# simulateAndPlot(structureMatrix, familyMatrix, sampleSize, 1:6, normal_corr_values, general_dep=-5, general_family=23, seed=1)

# Sample from Vine
vineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, sampleSize)
vineOutput$RVM
print(cor(vineOutput$simdata))
print(compute_standardized_precision_matrix(vineOutput$simdata))

npReparamVine <- simulateAndReparameterizeVine(
  structureMatrix, 
  familyMatrix, 
  parameterMatrix, 
  sampleSize, 
  1:D, 
  c(0.7)
)
oldVineOutput <- npReparamVine$oldVineOutput
newVineOutput <- npReparamVine$newVineOutput
margins <- oldVineOutput$simdata
margins_np <- newVineOutput$simdata

ggplot(tibble(v1=qnorm(oldVineOutput$simdata[, 2]), v6=qnorm(oldVineOutput$simdata[, 4])), aes(x=v1, y=v6)) + 
  # geom_point(alpha=0.2) + 
  geom_density2d_filled(alpha=0.6) +
  labs(title='True Marginal Dependence between V4 and V6')

ggplot(tibble(v1=qnorm(newVineOutput$simdata[, 2]), v6=qnorm(newVineOutput$simdata[, 4])), aes(x=v1, y=v6)) + 
  # geom_point(alpha=0.2) + 
  geom_density2d_filled(alpha=0.6) +
  labs(title='NP Marginal Dependence between V4 and V6')
 # We check that the partial correlations are indeed zero for V6 and the other variables.

# ggpairs(qnorm(margins_np))
# We know what the 56 margin should be (normal gaussian)
# Can compare that to the marginal of 45 (which should be clayton)
# Check that this is uniform for the true distribution and check if true for non-paranormal approx
F4_5_np <- BiCopHfunc(margins_np[,2], margins_np[,3], family=4, par=clayton_dep)$hfunc2
emp_corr <- cor(qnorm(margins_np[,4]), qnorm(margins_np[,3]))
F6_5_np <- BiCopHfunc(margins_np[,4], margins_np[,3], family=1, par=emp_corr)$hfunc2
cor.test(F4_5_np, F6_5_np, method=c("kendall"))

ggplot(data.frame(F5 = qnorm(margins_np[,3]), F6 = qnorm(margins_np[,4])), aes(x = F5, y = F6)) +
  geom_density2d_filled(alpha=0.6) +
  labs(x = "F5", y = "F6", title = "NP Contour Plot of F5 vs F6") +
  theme_minimal()
ggplot(data.frame(F5 = qnorm(margins[,3]), F6 = qnorm(margins[,4])), aes(x = F5, y = F6)) +
  geom_density2d_filled(alpha=0.6) +
  labs(x = "F5", y = "F6", title = "True Contour Plot of F5 vs F6") +
  theme_minimal()



# Generate example data
set.seed(123)
X <- qnorm(margins[, 2])
Y <- qnorm(margins[, 4])
Z <- qnorm(margins[, 3])

# Perform Kernel Conditional Independence Test
kci_result <- KCI(X, Y, Z)
print(kci_result)

X_np <- qnorm(margins_np[, 2])
Y_np <- qnorm(margins_np[, 4])
Z_np <- qnorm(margins_np[, 3])

# Perform Kernel Conditional Independence Test
kci_result_np <- KCI(X_np, Y_np, Z_np)
print(kci_result_np)


F4_5 <- BiCopHfunc(margins[,4], margins[,5], family=23, par=clayton_dep)$hfunc2
F6_5 <- BiCopHfunc(margins[,6], margins[,5], family=1, par=normal_corr)$hfunc2
cor.test(F4_5, F6_5, method=c("kendall"))


p1 <- ggplot(data.frame(F4_5 = qnorm(F4_5_np), F6_5 = qnorm(F6_5_np)), aes(x = F4_5, y = F6_5)) +
  geom_density2d_filled(alpha=0.6) +
  labs(x = "F4_5", y = "F6_5", title = "NP Contour Plot of F4_5 vs F6_5") +
  theme_minimal()

# Plot for the true distribution
p2 <- ggplot(data.frame(F4_5 = qnorm(F4_5), F6_5 = qnorm(F6_5)), aes(x = F4_5, y = F6_5)) +
  geom_density2d_filled(alpha=0.6) +
  labs(x = "F4_5", y = "F6_5", title = "True Contour Plot of F4_5 vs F6_5") +
  theme_minimal()

# Combine the plots side by side
grid.arrange(p1, p2, ncol = 2)

x <- pnorm(rnorm(sampleSize))
y <- pnorm(rnorm(sampleSize, rho*x, sqrt(1-rho^2)))
z <- pnorm(rnorm(sampleSize, rho*y, sqrt(1-rho^2)))
d <- data.frame(x=x,y=y,z=z)
Fx_y <- BiCopHfunc(x, y, family=1, par=rho)$hfunc2
Fz_y <- BiCopHfunc(z, y, family=1, par=rho)$hfunc2
ggplot(data.frame(Fz_y=Fz_y, Fx_y=Fx_y), aes(x = Fz_y, y = Fx_y)) +
  geom_density_2d() + 
  labs(x = "Fz_y", y = "Fx_y", title = "Contour Plot of Fz_y vs Fx_y") +
  theme_minimal()


# True Model 
trueStructureMatrix <- matrix(
  c(4, 6, 5,
    0, 5, 6,
    0, 0, 6), ncol=D)
trueFamilyMatrix <- matrix(
  c(0, 0, 23,
    0, 0, 1,
    0, 0, 0), ncol=D)
npStructureMatrix <- matrix(
  c(6, 5, 4,
    0, 5, 6,
    0, 0, 6), ncol=D)
npFamilyMatrix <- matrix(
  c(0, 0, 23,
    0, 0, 1,
    0, 0, 0), ncol=D)

clayton_dep <- -10
normal_corr <- 0.5
trueParameterMatrix <- matrix(
  c(0, 0, clayton_dep,
    0, 0, normal_corr,
    0, 0, 0), ncol=D)