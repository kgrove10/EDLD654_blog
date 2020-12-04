#read in data
library(tidyverse)
full_train <- read_csv("data/train.csv",
                       col_types = cols(.default = col_guess(), 
                                        calc_admn_cd = col_character()))  %>% 
  select(-classification)

#str(full_train)

set.seed(500)
# import fall membership report data and clean 

sheets <- readxl::excel_sheets("data/fallmembershipreport_20192020.xlsx")
ode_schools <- readxl::read_xlsx("data/fallmembershipreport_20192020.xlsx", sheet = sheets[4])
str(ode_schools)

ethnicities <- ode_schools %>% select(attnd_schl_inst_id = `Attending School ID`,
                                      sch_name = `School Name`,
                                      contains("%")) %>%
  janitor::clean_names()

names(ethnicities) <- gsub("x2019_20_percent", "p", names(ethnicities))

head(ethnicities)


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


#Join ethnicity data and training data
full_train <- left_join(full_train, ethnicities)

#dim(full_train)

# import tidied frl and student count data 
frl <- read_csv("data/frl_stucounts.csv")

# add frl data to train data
frl_fulltrain <- left_join(full_train, frl)

#dim(frl_fulltrain)

#split data and resample

library(tidymodels)

set.seed(7895)
edu_split <- initial_split(frl_fulltrain)
train <- training(edu_split)
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
cores <- 8

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

#create grid
rf_grid_reg <- grid_regular(
  mtry(c(5, 5)),
  trees(c(1323, 1323)),
  min_n(range = c(30, 60)),
  levels = c(1, 1, 20))
  
#fit the tuning model
rf_grid_res <- tune_grid(
  rf_tune_wflow,
  train_cv,
  grid = rf_grid_reg,
  metrics = metrics_eval, #from above - same metrics of rsq, rmse, huber_loss
  control = control_resamples(verbose = TRUE,
                              save_pred = TRUE,
                              extract = function(x) extract_model(x)))

saveRDS(rf_grid_res, "RFTuneTalapasGrid.Rds")

#collect metrics
rf_grid_met <- rf_grid_res %>%
  collect_metrics() 

rf_grid_rsq <- rf_grid_met %>%
  filter(.metric == "rsq") %>%
  arrange(.metric, desc(mean)) %>%
  slice(1:5)

rf_grid_rmse <- rf_grid_met %>%
  filter(.metric == "rmse") %>%
  arrange(.metric, mean) %>%
  slice(1:5)

rf_grid_hl <- rf_grid_met %>%
  filter(.metric == "huber_loss") %>%
  arrange(.metric, mean) %>%
  slice(1:5)

rf_grid_metrics <- rbind(rf_grid_rsq, rf_grid_rmse, rf_grid_hl) 

rf_grid_metrics %>%
  write.csv("./RFTuneMetricsGrid.csv", row.names = FALSE)

#look at plot of tuned metrics:
rf_grid_res %>%
  autoplot() +
  geom_line()

#save plot for use on blog:

ggsave("RFTunedMetricsGrid.pdf",
       plot = last_plot(),
       scale = 1)

#select best results, based on rmse:
rf_best <- select_best(rf_grid_res, metric = "rmse")

#finalize our work flow based on this best result:

rf_wf_final <- finalize_workflow(
  rf_tune_wflow,
  rf_best)


#make predictions on test.csv using this final workflow
full_test <- read_csv("data/test.csv",
                      col_types = cols(.default = col_guess(), 
                                       calc_admn_cd = col_character()))
#str(full_test)

#join with ethnicity data
full_test_eth <- left_join(full_test, ethnicities) #join with FRL dataset
#str(full_test_eth)

#join with frl
full_test_FRL <- left_join(full_test_eth, frl)

nrow(full_test_FRL)

#workflow
fit_workflow <- fit(rf_wf_final, frl_fulltrain)

#use model to make predictions for test dataset
preds_final <- predict(fit_workflow, full_test_FRL) #use model to make predictions for test dataset

saveRDS(preds_final, "RFPreds.Rds")

head(preds_final)

pred_frame <- tibble(Id = full_test_FRL$id, Predicted = preds_final$.pred)
head(pred_frame, 20)
nrow(pred_frame)

#create prediction file
write_csv(pred_frame, "fit_rf.csv")





  


