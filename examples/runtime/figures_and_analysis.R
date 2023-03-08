library(ggplot2)
library(tidyverse)

causl_data <- read_csv('causl_runtime_concatenated.csv') %>%
  arrange(sampler_name, model_type, N, rho)
python_data <- read_csv('frugalCopyla_runtime_results.csv') %>%
  filter(runtime > 0) %>%
  mutate(algo_runtime=if_else(runtime < 0, 0.01, runtime)) %>%
  mutate(runtime=(algo_runtime/min_ess) * N) %>%
  mutate(sampler_name='frugalCopyla')

total_results <- bind_rows(
  causl_data, python_data
)
didelez_examples <- total_results %>% filter(model_type == 'didelez')
trivar_gauss_examples <- total_results %>% filter(model_type == 'trivariate_gaussian')

aggregated_didelez <- didelez_examples %>%
  group_by(sampler_name, model_type, N) %>%
  summarise(mean_runtime=mean(runtime), max_runtime=max(runtime), min_runtime=min(runtime))

didelez_p <- aggregated_didelez %>%
  ggplot() +
  geom_line(aes(x=N, y=mean_runtime, col=sampler_name)) +
  geom_errorbar(aes(x=N, ymin=min_runtime, ymax=max_runtime, col=sampler_name)) +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') + 
  labs(
    # title='Didelez Example',
    x='Number of Samples',
    y='Runtime / s',
    col='Sampler Type'
  )
ggsave('figures/didelez_runtime.pdf', didelez_p, height=6, width=8)#, units=c('cm'))

aggregated_trivar_gauss <- trivar_gauss_examples %>%
  group_by(sampler_name, model_type, N, rho) %>%
  summarise(mean_runtime=mean(runtime), max_runtime=max(runtime), min_runtime=min(runtime)) 

trivar_p <- aggregated_trivar_gauss %>%
  ggplot() +
  geom_line(aes(x=N, y=mean_runtime, col=sampler_name)) +
  geom_errorbar(aes(x=N, ymin=min_runtime, ymax=max_runtime, col=sampler_name)) +
  facet_grid(~ rho) +
  scale_x_continuous(trans='log10') + 
  scale_y_continuous(trans='log10') + 
  labs(
    # title='Trivariate Gaussian',
    x='Number of Samples',
    y='Runtime / s',
    col='Sampler Type'
  )
ggsave('figures/trivar_gauss_runtime.pdf', trivar_p, height=4, width=10)#, units=c('cm'))