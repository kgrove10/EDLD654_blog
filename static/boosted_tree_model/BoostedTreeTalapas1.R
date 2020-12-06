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


library(tidymodels)
library(xgboost)

set.seed(500)
edu_split <- initial_split(frl_fulltrain)
train <- training(edu_split)
test <- testing(edu_split)
train_cv <- vfold_cv(train, strata = "score", v = 10)

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

set.seed(500)

#default model without tuning
mod <- boost_tree() %>% 
  set_engine("xgboost", nthreads = parallel::detectCores()) %>% 
  set_mode("regression") %>% 
  set_args(trees = 5000, #number of trees in the ensemble
           stop_iter = 20, #the number of iterations without improvement before stopping
           validation = 0.2,
           learn_rate = 0.1) #the rate at which boosting algoirithm adapts at each iteration


#create workflow for default boosted model
wf_df <- workflow() %>% 
  add_recipe(rec) %>% 
  add_model(mod)


#let's get a sense of what this tuned model looks like
translate(mod)


#fit model w/ tuned learning rate
tune_tree_lr <- fit_resamples(
  wf_df, 
  train_cv, 
  metrics = metric_set(rmse, rsq),
  control = control_resamples(verbose = TRUE,
                              save_pred = TRUE,
                              extract = function(x) extract_model(x)))

saveRDS(tune_tree_lr, "BTTuneTalapasGrid1.Rds")


#collect metrics
bt_grid_met <- tune_tree_lr %>%
  collect_metrics() 

bt_grid_met %>%
  write.csv("./BTTuneMetricsDefault.csv", row.names = FALSE)



#apply to test split
test_fit <- last_fit(wf_df, edu_split)
test_metrics <- test_fit$.metrics

test_metrics %>%
  write.csv("./BTTestMetrics1.csv", row.names = FALSE)

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
fit_workflow <- fit(wf_df, frl_fulltrain)

#use model to make predictions for test dataset
preds_final <- predict(fit_workflow, full_test_FRL) #use model to make predictions for test dataset

saveRDS(preds_final, "BTPreds.Rds")

head(preds_final)

pred_frame <- tibble(Id = full_test_FRL$id, Predicted = preds_final$.pred)
head(pred_frame, 20)
nrow(pred_frame)

#create prediction file
write_csv(pred_frame, "fit_bt.csv")
