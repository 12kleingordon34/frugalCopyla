library(copula)
library(ggplot2)
library(gridExtra)
library(ppcor)
library(VineCopula)

source('nonparanormal.R')

sampleSize <- 5000
D <- 6

# Making Clayton model
structureMatrix <- matrix(
                   c(1, 6, 5, 4, 3, 2,
                     0, 2, 6, 5, 4, 3,
                     0, 0, 3, 6, 5, 4,
                     0, 0, 0, 4, 6, 5,
                     0, 0, 0, 0, 5, 6,
                     0, 0, 0, 0, 0, 6), ncol=D)
familyMatrix <- matrix(
                c(0, 0, 0, 0, 0, 23, 
                  0, 0, 0, 0, 0, 23,
                  0, 0, 0, 0, 0, 23,
                  0, 0, 0, 0, 0, 23,
                  0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0, 0), ncol=D)

clayton_dep <- -5
normal_corr <- 0.5
parameterMatrix <- matrix(
                   c(0, 0, 0, 0, 0, clayton_dep, 
                     0, 0, 0, 0, 0, clayton_dep,
                     0, 0, 0, 0, 0, clayton_dep,
                     0, 0, 0, 0, 0, clayton_dep,
                     0, 0, 0, 0, 0, normal_corr,
                     0, 0, 0, 0, 0, 0), ncol=D)

normal_corr_values <- seq(0.38, 0.98, by = 0.2)
simulateAndPlot(structureMatrix, familyMatrix, sampleSize, 1:6, normal_corr_values, general_dep=-5, general_family=23, seed=1)

# Sample from Vine
vineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, sampleSize)
gaussCop <- normalCopula(dim = D, dispstr = "un")

# Fit the MVG copula model to the data using MLE
fit <- fitCopula(gaussCop, data = vineOutput$simdata, method = method)

vineCopFit <- RVineStructureSelect(data = vineOutput$simdata, familyset = 1)

mvgFit <- fitMVGaussianCopula(vineOutput$simdata, method='itau')
contour(vineOutput$RVM)

# Settings for reparameterisation
topoOrder=1:D
corMatrixMN=mvgFit$correlationMatrix[1:(D-1), 1:(D-1)]
vineCorParams=normal_corr

npReparamVine <- simulateAndReparameterizeVine(
  structureMatrix, 
  familyMatrix, 
  parameterMatrix, 
  sampleSize, 
  topoOrder, 
  vineCorParams
)
oldVineOutput <- npReparamVine$oldVineOutput
newVineOutput <- npReparamVine$newVineOutput
print(npReparamVine$standardized_precision_matrix_np)
contour(newVineOutput$RVM)
margins <- oldVineOutput$simdata
margins_np <- newVineOutput$simdata

# We check that the partial correlations are indeed zero for V6 and the other variables.

# ggpairs(qnorm(margins_np))
# We know what the 56 margin should be (normal gaussian)
# Can compare that to the marginal of 45 (which should be clayton)
# Check that this is uniform for the true distribution and check if true for non-paranormal approx
F4_5_np <- BiCopHfunc(margins_np[,4], margins_np[,5], family=23, par=clayton_dep)$hfunc2
F6_5_np <- BiCopHfunc(margins_np[,6], margins_np[,5], family=1, par=normal_corr)$hfunc2

F4_5 <- BiCopHfunc(margins[,4], margins[,5], family=23, par=clayton_dep)$hfunc2
F6_5 <- BiCopHfunc(margins[,6], margins[,5], family=1, par=normal_corr)$hfunc2

p1 <- ggplot(data.frame(F4_5 = F4_5_np, F6_5 = F6_5_np), aes(x = F4_5, y = F6_5)) +
  geom_density_2d() +
  labs(x = "F4_5", y = "F6_5", title = "NP Contour Plot of F4_5 vs F6_5") +
  theme_minimal()

# Plot for the true distribution
p2 <- ggplot(data.frame(F4_5 = F4_5, F6_5 = F6_5), aes(x = F4_5, y = F6_5)) +
  geom_density_2d() +
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