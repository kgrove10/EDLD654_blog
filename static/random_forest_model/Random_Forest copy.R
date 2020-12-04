#read in data
library(tidyverse)
full_train <- read_csv("data/train.csv",
                       col_types = cols(.default = col_guess(), 
                                        calc_admn_cd = col_character()))  %>% 
  select(-classification)

#str(full_train)

library(rio)
set.seed(500)
# import fall membership report data and clean 

sheets <- readxl::excel_sheets("data/fallmembershipreport_20192020.xlsx")
ode_schools <- readxl::read_xlsx(here::here("data/fallmembershipreport_20192020.xlsx"), sheet = sheets[4])
str(ode_schools)

ethnicities <- ode_schools %>% select(attnd_schl_inst_id = `Attending School ID`,
                                      sch_name = `School Name`,
                                      contains("%")) %>%
  janitor::clean_names()

names(ethnicities) <- gsub("x2019_20_percent", "p", names(ethnicities))

head(ethnicities)

#function to check id is unique
unique_id <- function(x, ...) {
  id_set <- x %>% select(...)
  id_set_dist <- id_set %>% distinct
  if (nrow(id_set) == nrow(id_set_dist)) {
    TRUE
  } else {
    non_unique_ids <- id_set %>% 
      filter(id_set %>% duplicated()) %>% 
      distinct()
    suppressMessages(
      inner_join(non_unique_ids, x) %>% arrange(...)
    )
  }
}

#make ethnicities have attnd_schl_inst_id as a unique identifier - clunky, but works!
ethnicities <- ethnicities %>%
  group_by(attnd_schl_inst_id) %>%
  summarize(p_american_indian_alaska_native = mean(p_american_indian_alaska_native),
            p_asian = mean(p_asian),
            p_native_hawaiian_pacific_islander = mean(p_native_hawaiian_pacific_islander),
            p_black_african_american = mean(p_black_african_american),
            p_hispanic_latino = mean(p_hispanic_latino),
            p_white = mean(p_white),
            p_multiracial = mean(p_multiracial))

#more concise code that should work, but doesn't because of difficulties passing dataframe to mean fct
#ethnicities <- ethnicities %>% 
#group_by(attnd_schl_inst_id) %>% 
#summarise(across(everything(), list(mean)))

ethnicities %>% unique_id(attnd_schl_inst_id)

#Join ethnicity data and training data
full_train <- left_join(full_train, ethnicities)

dim(full_train)

# import and tidy free and reduced lunch data 
frl <- rio::import("https://nces.ed.gov/ccd/Data/zip/ccd_sch_033_1718_l_1a_083118.zip",
                   setclass = "tbl_df")  %>% 
  janitor::clean_names()  %>% 
  filter(st == "OR")  %>%
  select(ncessch, lunch_program, student_count)  %>% 
  mutate(student_count = replace_na(student_count, 0))  %>% 
  pivot_wider(names_from = lunch_program,
              values_from = student_count)  %>% 
  janitor::clean_names()  %>% 
  mutate(ncessch = as.double(ncessch))

# import student counts for each school across grades
stu_counts <- rio::import("https://github.com/datalorax/ach-gap-variability/raw/master/data/achievement-gaps-geocoded.csv", setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))

# join frl and stu_counts data
frl <- left_join(frl, stu_counts)

#save as .csv for use on talapas
write_csv(frl, "data/frl_stucounts.csv")

# add frl data to train data
frl_fulltrain <- left_join(full_train, frl)

dim(frl_fulltrain)

getwd()
#look at relations between variables
#jpeg("VariableRelations.jpg")

var_relate <- frl_fulltrain %>% 
  select(-contains("id"), -ncessch, -missing, -not_applicable) %>% 
  select_if(is.numeric) %>% 
  select(score, everything()) %>% 
  cor(use = "complete.obs") %>% 
  corrplot::corrplot()

#dev.off()

dev.copy(pdf,'plots/VariableRelations.pdf')
dev.off()

#split data and resample

library(tidymodels)

set.seed(7895)
edu_split <- initial_split(frl_fulltrain)
train <- training(edu_split) %>%
  dplyr::sample_frac(0.005)
test <- testing(edu_split)
train_cv <- vfold_cv(train, strata = "score", v = 2)

str(train_cv)

#create recipe and prep
rec <- recipe(score ~., train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt),
              time_index = as.numeric(tst_dt)) %>%
  step_rm(contains("bnch")) %>%
  update_role(tst_dt, new_role = "time_index") %>%
  update_role(contains("id"), ncessch, new_role = "id") %>% #removed sch_name
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_unknown(all_nominal(), -all_outcomes()) %>%
  step_rollimpute(all_numeric(), -all_outcomes(), -has_role("id"), -n) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id")) %>%
  step_nzv(all_predictors(), freq_cut = 0, unique_cut = 0) %>%
  step_dummy(all_nominal(), -has_role(match = "id"), -all_outcomes(), -time_index) %>%
  step_nzv(all_predictors()) %>%
  step_interact(terms = ~ starts_with('lat'):starts_with("lon"))

prep(rec)

#create random forest tuning model
cores <- parallel::detectCores()
cores

rf_def_mod <- rand_forest() %>%
  set_engine("ranger",
             num.threads = cores,
             importance = "permutation",
             verbose = TRUE) %>%
  set_mode("regression")

rf_tune_mod <- rf_def_mod %>%
  set_args(
    mtry = tune(),
    trees = tune(),
    min_n = tune()
  )

translate(rf_tune_mod)

#create workflow, set metrics
rf_tune_wflow <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_tune_mod)

metrics_eval <- metric_set(rmse, 
                           rsq, 
                           huber_loss)

#fit the tuning model
tictoc::tic()

rf_tune_res <- tune_grid(
  rf_tune_wflow,
  train_cv,
  tune = 20,
  metrics = metrics_eval, #from above - same metrics of rsq, rmse, huber_loss
  control = control_resamples(verbose = TRUE,
                              save_pred = TRUE,
                              extract = function(x) extract_model(x)))

tictoc::toc()

#collect metrics
rf_tune_met <- rf_tune_res %>%
  collect_metrics() 

rf_tune_rsq <- rf_tune_met %>%
  filter(.metric == "rsq") %>%
  arrange(.metric, desc(mean)) %>%
  slice(1:5)

rf_tune_rmse <- rf_tune_met %>%
  filter(.metric == "rmse") %>%
  arrange(.metric, mean) %>%
  slice(1:5)

rf_tune_hl <- rf_tune_met %>%
  filter(.metric == "huber_loss") %>%
  arrange(.metric, mean) %>%
  slice(1:5)

rf_tune_metrics <- rbind(rf_tune_rsq, rf_tune_rmse, rf_tune_hl) 

rf_tune_metrics %>%
  write.csv("./RFTuneMetrics.csv", row.names = FALSE)

#look at plot of tuned metrics:
rf_tune_res %>%
  autoplot() +
  geom_line()

#save plot for use on blog:

ggsave("RFTunedMetrics.pdf",
       plot = last_plot(),
       scale = 1)

#based on these plots, it looks like the min_n is best just below 30, where the min_n and rmse are minimized, and the rsq is maximized. 
# I'm going to examine the values more closely between 20 and 35.
# for the # of trees, the best point seems to be around 600(ish - rough reading from graph).  Going to examine between 500 and 1000
# for mtry, going to try between 0 and... nevermind, seems to be doing a good job!

#select best results, based on rmse:

rf_best <- select_best(rf_tune_res, metric = "rmse")

#finalize our work flow based on this best result:

rf_wf_final <- finalize_workflow(
  rf_tune_wflow,
  rf_best)


#make predictions on test.csv using this final workflow
full_test <- read_csv("test.csv",
                      col_types = cols(.default = col_guess(), 
                                       calc_admn_cd = col_character()))
#str(full_test)

#join with ethnicity data
full_test_eth <- left_join(full_test, ethnicities) #join with FRL dataset
#str(full_test_eth)

#join with frl
full_test_FRL <- left_join(full_test_eth, frl) %>%
  dplyr::sample_frac(0.005)

nrow(full_test_FRL)

#workflow
fit_workflow <- fit(rf_wf_final, frl_fulltrain)

#use model to make predictions for test dataset
preds_final <- predict(fit_workflow, full_test_FRL) #use model to make predictions for test dataset

head(preds_final)

pred_frame <- tibble(Id = full_test_FRL$id, Predicted = preds_final$.pred)
head(pred_frame, 20)
nrow(pred_frame)

#create prediction file
write_csv(pred_frame, "fit_rf.csv")


