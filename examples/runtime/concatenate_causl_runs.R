library(tidyverse)

data_paths <- list.files(path='./data/')

results <- tibble()
for (path in data_paths) {
  results <- bind_rows(
    results,
    suppressMessages(read_csv(paste0('./data/', path)))
  )
}

write_csv(results, 'causl_runtime_concatenated.csv')