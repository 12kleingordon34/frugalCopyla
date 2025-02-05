library(copula)
library(ggplot2)
library(gridExtra)
library(tidyverse)
library(ppcor)
library(VineCopula)

source('nonparanormal.R')

sampleSize <- 200
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


num_bootstraps = 10
normal_corr_range = seq(0.1, 0.9, 0.1)

results <- tibble(
  norm_corr=numeric(),
  bootstrap_no=numeric(),
  method=character(),
  pval=numeric()
)
for (b in 1:num_bootstraps) {
  for (normal_corr in normal_corr_range) {
    print(paste0('Normal Corr: ', normal_corr, '. Run: ', b, ' / ', num_bootstraps))
    parameterMatrix <- matrix(
      c(0, 0, 0, 0, 0, clayton_dep, 
        0, 0, 0, 0, 0, clayton_dep,
        0, 0, 0, 0, 0, clayton_dep,
        0, 0, 0, 0, 0, clayton_dep,
        0, 0, 0, 0, 0, normal_corr,
        0, 0, 0, 0, 0, 0), ncol=D)
    
    # Sample from Vine
    vineOutput <- simulateRVineData(structureMatrix, familyMatrix, parameterMatrix, sampleSize, seed=b)
    gaussCop <- normalCopula(dim = D, dispstr = "un")
    
    # Fit the MVG copula model to the data using MLE
    fit <- fitCopula(gaussCop, data = vineOutput$simdata, method = 'itau')
    vineCopFit <- RVineStructureSelect(data = vineOutput$simdata, familyset = 1)
    mvgFit <- fitMVGaussianCopula(vineOutput$simdata, method='itau')

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
      vineCorParams,
      seed=b
    )
    oldVineOutput <- npReparamVine$oldVineOutput
    newVineOutput <- npReparamVine$newVineOutput
    margins <- oldVineOutput$simdata
    margins_np <- newVineOutput$simdata
    
    # We check that the partial correlations are indeed zero for V6 and the other variables.
    
    # ggpairs(qnorm(margins_np))
    # We know what the 56 margin should be (normal gaussian)
    # Can compare that to the marginal of 45 (which should be clayton)
    # Check that this is uniform for the true distribution and check if true for non-paranormal approx
    F4_5_np <- BiCopHfunc(margins_np[,4], margins_np[,5], family=23, par=clayton_dep)$hfunc2
    F6_5_np <- BiCopHfunc(margins_np[,6], margins_np[,5], family=1, par=normal_corr)$hfunc2
    test_np = cor.test(F4_5_np, F6_5_np, method=c("kendall"))
    pval_np = test_np$p.value
    
    F4_5 <- BiCopHfunc(margins[,4], margins[,5], family=23, par=clayton_dep)$hfunc2
    F6_5 <- BiCopHfunc(margins[,6], margins[,5], family=1, par=normal_corr)$hfunc2
    test_true = cor.test(F4_5, F6_5, method=c("kendall"))
    pval_true = test_true$p.value
    results <- rbind(
      results,
      rbind(
        tibble(
          norm_corr=normal_corr,
          bootstrap_no=b,
          method='Nonparanormal',
          pval=pval_np
        ),
        tibble(
          norm_corr=normal_corr,
          bootstrap_no=b,
          method='True',
          pval=pval_true
        )
      )
    )
  }
}
summary_data <- results %>%
  group_by(norm_corr, method) %>%
  summarise(
    mean_pval = mean(pval, na.rm = TRUE),
    sd_pval = sd(pval, na.rm = TRUE)
  ) %>%
  mutate(error = 2 * sd_pval)

# Create the plot
ggplot(summary_data, aes(x = norm_corr, y = mean_pval, color = method)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = mean_pval - error, ymax = mean_pval + error), width = 0.1) +
  labs(title = "Average p-values with 2*std Error Bars",
       x = "Norm Corr",
       y = "Mean P-value",
       color = "Method") +
  theme_minimal()

ggplot(results, aes(x = norm_corr, y = pval, color = as.factor(bootstrap_no))) +
  geom_line() +
  facet_wrap(~ method) +
  labs(title = "P-values vs Norm Corr by Method and Bootstrap",
       x = "Norm Corr",
       y = "P-value",
       color = "Bootstrap No") +
  theme_minimal()
