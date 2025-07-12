library(tidyverse)
mturk_list <- read_csv("data/blank_images.csv", show_col_types = FALSE) %>%
  separate_wider_regex(file_name, 
                       patterns = c("cell_", legajo = "\\d+", "_", col = "[\\w_]+?", "_", row = "\\d+", "\\.png"),
                       cols_remove = FALSE) %>%
  select(-date, -time) %>%
  pivot_wider(id_cols = c(legajo, col), 
              names_from = row, 
              values_from = file_name,
              names_prefix = "row_")
write_csv(mturk_list, "output/mturk_list.csv")
