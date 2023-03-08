library(ggplot2)
library(tidyverse)

dir_loc <- './data/'

causl_paths = list.files(
  path=dir_loc,
  pattern='results_row'
)

concatenated_results = tibble()

for (path in causl_paths) {
  concatenated_results = bind_rows(
    concatenated_results,
    suppressMessages(read_csv(paste0(dir_loc, path)))
  )
}
concatenated_results <- concatenated_results %>% rename(run_idx=idx)

python_data = read_csv('frugalCopyla_runtime_results.csv') %>% 
  mutate(sampler_name='frugalCopyla')
concatenated_results <- bind_rows(
  concatenated_results,
  python_data
)
write_csv(concatenated_results, "full_runtime_results.csv")

#### FIGURES ####
concatenated_results %>% 
  group_by(sampler_name, model_type, N, rho) %>%
  summarise(m_runtime=mean(runtime), sd_runtime=sd(runtime)) %>%
  filter(model_type == 'didelez') %>%
ggplot() +
  geom_line(aes(y=m_runtime, x=N, color=sampler_name)) + 
  geom_errorbar(aes(ymin=m_runtime-2*sd_runtime, ymax=m_runtime+2*sd_runtime, x=N, color=sampler_name)) +
  scale_x_log10() +
  labs(
    title='Runtime to generate N samples from the Evans and Didelez Example',
    x='N (Simulated Dataset Size)',
    y='Runtime / s',
    color='Sampler type'
  )
  
concatenated_results %>% 
  group_by(sampler_name, model_type, N, rho) %>%
  summarise(m_runtime=mean(runtime), sd_runtime=sd(runtime)) %>%
  filter(model_type == 'trivariate_gaussian') %>%
  ggplot() +
  geom_line(aes(y=m_runtime, x=N, color=sampler_name)) + 
  geom_errorbar(aes(ymin=m_runtime-2*sd_runtime, ymax=m_runtime+2*sd_runtime, x=N, color=sampler_name)) +
  facet_grid(~ rho) + 
  scale_y_log10() +
  scale_x_log10() +
  labs(
    title='Runtime to generate N samples from a Correlated Trivariate Gaussian',
    x='N (Simulated Dataset Size)',
    y='Runtime / s',
    color='Sampler type'
  )
