# Model Training----------------------
## Random forest ---------------------------
train_ratio <- commandArgs(trailingOnly = TRUE)

library(dplyr)
library(randomForest)
library(caret)
library(doParallel)

data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = T
)

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index

data_train <- data_df[train_index, ]
data_test <- data_df[test_index, ]

tuning_grid = expand.grid(mtry = seq(4, 10, 1))

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
modellist <- list()


type <- ifelse(.Platform$OS.type == 'windows', 'PSOCK', 'FORK')
cl <- makeCluster(16, type)
registerDoParallel(cores = cl)

# train with different ntree parameters
for (ntree in seq(200, 1200, 100)) {
  set.seed(123)
  fit <- train(
    outcome ~ .,
    data = data_train,
    method = 'rf',
    metric = 'ROC',
    tuneGrid = tuning_grid,
    trControl = ctrl,
    ntree = ntree
  )
  key <- toString(ntree)
  modellist[[key]] <- fit
}

stopCluster(cl)

result_file <- paste0(
  '../Results/model_fit_binary_outcome_version8_features_filtered/rf_ROC_vary_mtry_ntree_train_ratio_',
  train_ratio,
  '.rds'
)
saveRDS(modellist, file = result_file)

### Results integration-------------
library(dplyr);require(randomForest)

data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')
loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = T
)
index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

measure_mat_ls <- list()
var_imp_ls <- list()
prob_mat_ls <- list()

for(train_ratio in c(0.6, 0.7, 0.8)) {
  result_file <- paste0(
    '../Results/model_fit_binary_outcome_version8_features_filtered/rf_ROC_vary_mtry_ntree_train_ratio_',
    train_ratio,
    '.rds'
  )
  modellist <- readRDS(result_file)
  
  train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
  test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index
  
  data_train <- data_df[train_index, ]
  data_test <- data_df[test_index, ]
  
  results_modells <- resamples(modellist)
  sum.results_model_ls <- summary(results_modells)
  best_ntree <- names(which.max(sum.results_model_ls$statistics$ROC[, 'Mean']))
  
  best_rf_tuned <- modellist[[best_ntree]]
  best_mtry <- best_rf_tuned$bestTune
  
  prob_train <- predict(best_rf_tuned, type = 'prob')[, 2]
  roc_train <- roc(data_train$outcome, prob_train)
  auc_train <- auc(roc_train)
  best_cutoff_train <- coords(roc_train, 'best')
  pred_class_train <- factor(
    ifelse(prob_train > as.numeric(best_cutoff_train[1]), 'birth', 'fail'),
    levels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_train <- confusionMatrix(pred_class_train, data_train$outcome)
  measure_train <- c(auc_train,
                     conf_matrix_train$overall[c('Accuracy', 'Kappa')],
                     conf_matrix_train$byClass)
  names(measure_train)[1] <- 'Auc'
  
  measure_mat <- NULL
  measure_mat <- rbind(measure_mat, measure_train)
  
  
  prob_test <- predict(best_rf_tuned, newdata = data_test, type = "prob")[, 2]
  
  library(pROC)
  roc_obj_test <- roc(data_test$outcome, prob_test)
  auc_test <- auc(roc_obj_test)
  
  best_cutoff_test <- coords(roc_obj_test, "best")
  pred_type_test <- factor(
    ifelse(prob_test > as.numeric(best_cutoff_test[1]), 'birth', 'fail'),
    levels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_test <- confusionMatrix(pred_type_test, data_test$outcome)
  
  measure_test <- c(auc_test,
                    conf_matrix_test$overall[c('Accuracy', 'Kappa')],
                    conf_matrix_test$byClass)
  names(measure_test)[1] <- 'Auc'
  
  cat(
    paste0(
      'train_ratio: ',
      train_ratio,
      ', best_mtry: ',
      best_mtry,
      ', best_ntree: ',
      best_ntree,
      '.\n'
    )
  )
  print(measure_test)
  
  measure_mat <- rbind(measure_mat, measure_test)
  
  set.seed(200)
  fit.rf <- randomForest(
    outcome ~ .,
    data = data_train,
    mtry = as.numeric(best_mtry),
    ntree = as.numeric(best_ntree)
  )
  
  prob_train_refit <- predict(fit.rf, type = 'prob')[, 2]
  roc_obs_train_refit <- roc(data_train$outcome, prob_train_refit)
  auc_score_train_refit <- auc(roc_obs_train_refit)
  
  best_cutoff_train_refit <- coords(roc_obs_train_refit, "best")
  
  pred_type_train_refit <- factor(
    ifelse(
      prob_train_refit > as.numeric(best_cutoff_train_refit[1]),
      'birth',
      'fail'
    ),
    levels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_train_refit <- confusionMatrix(pred_type_train_refit, data_train$outcome)
  measure_train_refit <- c(
    auc_score_train_refit,
    conf_matrix_train_refit$overall[c('Accuracy', 'Kappa')],
    conf_matrix_train_refit$byClass
  )
  names(measure_train_refit)[1] <- 'Auc'
  
  
  prob_test_refit <- predict(fit.rf, newdata = data_test, type = 'prob')[, 2, drop =
                                                                           T]
  roc_obs_test_refit <- roc(data_test$outcome, prob_test_refit)
  auc_score_test_refit <- auc(roc_obs_test_refit)
  
  var_imp.refit <- varImp(fit.rf)
  var_imp_refit <- data.frame(variables = row.names(var_imp.refit),
                              importance = var_imp.refit$birth)
  
  best_cutoff_test_refit <- coords(roc_obs_test_refit, "best")
  pred_type_refit <- factor(
    ifelse(
      prob_test_refit > as.numeric(best_cutoff_test_refit[1]),
      'birth',
      'fail'
    ),
    levels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_test_refit <- confusionMatrix(pred_type_refit, data_test$outcome)
  measure_test_refit <- c(
    auc_score_test_refit,
    conf_matrix_test_refit$overall[c('Accuracy', 'Kappa')],
    conf_matrix_test_refit$byClass
  )
  names(measure_test_refit)[1] <- 'Auc'
  
  
  measure_mat <- rbind(measure_mat, measure_train_refit, measure_test_refit)
  rownames(measure_mat) <- c('train', 'test', 'train_refit', 'test_refit')
  measure_mat <- cbind(mtry = as.numeric(best_mtry),
                       ntree = as.numeric(best_ntree),
                       measure_mat)
  
  
  var_imp_ls[[toString(train_ratio)]] <- var_imp_refit
  measure_mat_ls[[toString(train_ratio)]] <- measure_mat
  
  
  prob_ls <- list(
    outcome_train = data_train$outcome,
    outcome_test = data_test$outcome,
    prob_train = prob_train,
    prob_test = prob_test,
    prob_train_refit = prob_train_refit,
    prob_test_refit = prob_test_refit
  )
  
  
  prob_mat_ls[[toString(train_ratio)]] <- prob_ls
  
  
}


var_imp_comb <- data.frame(
  variables = var_imp_ls[['0.6']]$variables,
  impor_train_06 = var_imp_ls[['0.6']]$importance,
  impor_train_07 = var_imp_ls[['0.7']]$importance,
  impor_train_08 = var_imp_ls[['0.8']]$importance
)


write.csv(var_imp_comb, file = '../Results/Var_importance_binary_outcome_version8_features_filtered_randomforest.csv', row.names = F)


measure_df <- do.call(rbind, measure_mat_ls) %>%
  as.data.frame() %>%
  mutate(train_raio = rep(c(0.6, 0.7, 0.8), each = 4),
         data_type = rep(c(
           'train', 'test', 'train_refit', 'test_refit'
         ), times = 3)) %>%
  dplyr::select(train_raio, data_type, everything())


write.csv(measure_df, file = '../Results/model_performace_binary_outcome_version8_features_filtered_randomforest.csv', row.names = F)


saveRDS(prob_mat_ls, file = '../Results/prob_pred_binary_version8_features_filtered/prob_rf_ls.rds')

## XGBoost------------------
arg_vec <- as.numeric(commandArgs(trailingOnly = TRUE))

train_ratio <- arg_vec[1]
nrounds <- arg_vec[2]
max_depth <- arg_vec[3]
eta <- arg_vec[4]
gamma <- arg_vec[5]
colsample_bytree <- arg_vec[6]
min_child_weight <- arg_vec[7]


library(dplyr)
require(xgboost)
require(plyr)
library(caret)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index

data_train <- data_df[train_index, ]
data_test <- data_df[test_index, ]


X_train <- model.matrix(outcome ~ -1 + ., data = data_train)
y_train <- data_train$outcome
cv.fit.xgb <- xgb.cv(
  data = X_train,
  label = y_train,
  booster = "gbtree",
  objective = "binary:logistic",
  metrics = 'auc',
  nfold = 5,
  stratified = TRUE,
  nrounds = nrounds,
  max_depth = max_depth,
  colsample_bytree = colsample_bytree,
  min_child_weight = 1,
  subsample = 0.7,
  eta = eta,
  gamma = gamma,
  verbose = 0,
  early_stopping_rounds = 5
)

best_auc <- max(cv.fit.xgb$evaluation_log$test_auc_mean)
best_iter <- cv.fit.xgb$best_iteration


result_file <- paste0(
  '../Results/model_fit_binary_outcome_version8_features_filtered/cv_fit_xgboost/',
  'train_ratio_',
  train_ratio,
  '_nrounds_',
  nrounds,
  '_max_depth_',
  max_depth,
  '_eta_',
  eta,
  '_gamma_',
  gamma,
  '_colsample_bytree_',
  colsample_bytree,
  '_min_child_weight_',
  min_child_weight,
  '.rds'
)

saveRDS(list(
  cv.fit.xgb = cv.fit.xgb,
  best_auc = best_auc,
  best_iter = best_iter
),
result_file)



### Results integration-------------
library(ggplot2)
library(caret)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

measure_mat_ls <- list()
confu_mat_ls <- list()

prob_mat_ls <- list()

for (train_ratio in c(0.6, 0.7, 0.8)) {
  result_cv_mat <- NULL
  
  for (nrounds in c(1800, 2000, 2200, 2400)) {
    for (max_depth in c(9, 12, 15, 18)) {
      for (eta in c(0.01, 0.05)) {
        for (gamma in c(0.01, 0.05)) {
          for (colsample_bytree in c(0.4, 0.5, 0.6)) {
            for (min_child_weight in c(1)) {
              result_file <- paste0(
                '../Results/model_fit_binary_outcome_version8_features_filtered/cv_fit_xgboost/',
                'train_ratio_',
                train_ratio,
                '_nrounds_',
                nrounds,
                '_max_depth_',
                max_depth,
                '_eta_',
                eta,
                '_gamma_',
                gamma,
                '_colsample_bytree_',
                colsample_bytree,
                '_min_child_weight_',
                min_child_weight,
                '.rds'
              )
              
              result_ls <- readRDS(result_file)
              result_cv_mat <- rbind(
                result_cv_mat,
                c(
                  train_ratio,
                  nrounds,
                  max_depth,
                  eta,
                  gamma,
                  colsample_bytree,
                  min_child_weight,
                  result_ls$best_auc,
                  result_ls$best_iter
                )
              )
              
            }
          }
        }
      }
    }
  }
  colnames(result_cv_mat) <- c(
    "train_ratio",
    "nrounds",
    "max_depth",
    "eta",
    "gamma",
    "colsample_bytree",
    "min_child_weight",
    'auc_best',
    'iter_best'
  )
  
  result_cv_df <- as.data.frame(result_cv_mat)
  
  result_df <- result_cv_df
  result_df$eta  <- factor(result_df$eta,
                           levels = c(0.05, 0.01),
                           labels = paste0('eta=', c(0.05, 0.01)))
  result_df$gamma  <- factor(result_df$gamma,
                             levels = c(0.01, 0.05),
                             labels = paste0('gamma=', c(0.01, 0.05)))
  result_df$max_depth  <- factor(
    result_df$max_depth,
    levels = c(9, 12, 15, 18),
    labels = paste0('max_depth=', c(9, 12, 15, 18))
  )
  result_df$colsample_bytree  <- factor(
    result_df$colsample_bytree,
    levels = c(0.4, 0.5, 0.6),
    labels = paste0('colsample_bytree=', c(0.4, 0.5, 0.6))
  )
  
  
  
  loc_best <- which.max(result_cv_df$auc_best)
  
  train_ratio <- result_cv_df$train_ratio[loc_best]
  train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
  test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index
  
  data_train <- data_df[train_index, ]
  data_test <- data_df[test_index, ]
  
  
  library(caret)
  library(pROC)
  require(xgboost)
  
  X_train <- model.matrix(outcome ~ -1 + ., data = data_train)
  y_train <- data_train$outcome
  
  
  
  fit.xbg.best <- xgboost(
    data = X_train,
    label = y_train,
    booster = "gbtree",
    objective = "binary:logistic",
    nrounds = result_cv_df$nrounds[loc_best],
    max_depth = result_cv_df$max_depth[loc_best],
    colsample_bytree = result_cv_df$colsample_bytree[loc_best],
    min_child_weight = 1,
    subsample = 0.7,
    eta = result_cv_df$eta[loc_best],
    gamma = result_cv_df$gamma[loc_best],
    verbose = 0,
    early_stopping_rounds = 5
  )
  
  
  prob_train <- predict(fit.xbg.best, newdata = X_train, type = "response")
  
  # Evaluation Metrics for training data
  roc_obj_train <- roc(y_train, prob_train)
  auc_train <- auc(roc_obj_train)
  
  best_cutoff_train <- coords(roc_obj_train, "best")
  
  pred_class_train <- factor(ifelse(prob_train > as.numeric(best_cutoff_train[1]), 1, 0))
  conf_matrix_train <- confusionMatrix(pred_class_train, factor(data_train$outcome))
  
  measure_train <- c(auc_train,
                     conf_matrix_train$overall[c('Accuracy', 'Kappa')],
                     conf_matrix_train$byClass)
  names(measure_train)[1] <- 'Auc'
  
  
  measure_mat <- NULL
  measure_mat <- rbind(measure_mat, measure_train)
  
  
  X_test <- model.matrix(outcome ~ -1 + ., data = data_test)
  y_test <- data_test$outcome
  
  prob_test <- predict(fit.xbg.best, newdata = X_test, type = "response")
  
  # Evaluation Metrics for testing data
  roc_obj_test <- roc(y_test, prob_test)
  auc_test <- auc(roc_obj_test)
  
  best_cutoff_test <- coords(roc_obj_test, "best")
  
  pred_class_test <- ifelse(prob_test > as.numeric(best_cutoff_test[1]), 1, 0)
  conf_matrix_test <- confusionMatrix(factor(pred_class_test), factor(y_test))
  
  measure_test <- c(auc_test,
                    conf_matrix_test$overall[c('Accuracy', 'Kappa')],
                    conf_matrix_test$byClass)
  names(measure_test)[1] <- 'Auc'
  
  measure_mat <- rbind(measure_mat, measure_test)
  rownames(measure_mat) <- c('train', 'test')
  print(measure_mat)
  
  measure_mat_ls[[toString(train_ratio)]] <- measure_mat
  
  prob_ls <- list(
    outcome_train = y_train,
    outcome_test = y_test,
    prob_train = prob_train,
    prob_test = prob_test
  )
  
  
  prob_mat_ls[[toString(train_ratio)]] <- prob_ls
  
  
  
}

measure_df <- do.call(rbind, measure_mat_ls) %>%
  as.data.frame() %>%
  mutate(train_raio = rep(c(0.6, 0.7, 0.8), each = 2),
         data_type = rep(c('train', 'test'), times = 3)) %>%
  dplyr::select(train_raio, data_type, everything())



write.csv(measure_df, file = '../Results/model_performace_binary_outcome_version8_features_filtered_xgboost.csv', row.names = F)

saveRDS(prob_mat_ls, file = '../Results/prob_pred_binary_version8_features_filtered/prob_xgboost_ls.rds')


## LightGBM--------------------------
arg_vec <- commandArgs(trailingOnly = TRUE)

train_ratio <- arg_vec[1]
grid_row_start <- as.numeric(arg_vec[2])
total_size <- as.numeric(arg_vec[3])


library(dplyr)
library(tidymodels)
library(bonsai)

data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1
data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = T
)

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

train_index <- index_train_test_ls[[train_ratio]]$train_index
test_index <- index_train_test_ls[[train_ratio]]$test_index

data_train <- data_df[train_index, ]
data_test <- data_df[test_index, ]

bt_light <- boost_tree(
  trees = tune(),
  mtry = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_engine("lightgbm", objective = "multiclass", num_class = 2) %>%
  set_mode("classification")

bt_wf <- workflow() %>%
  add_formula(outcome ~ .) %>%
  add_model(bt_light)

set.seed(123)
tree_grid <- grid_max_entropy(
  trees(range = c(100, 1000)),
  mtry(range = c(3L, 8L)),
  tree_depth(range = c(3, 15)),
  learn_rate(),
  min_n(range = c(10, 40)),
  loss_reduction(),
  size = total_size
)


set.seed(123)
bt_folds <- vfold_cv(data_train, v = 5, strata = 'outcome')

set.seed(123)
library(doParallel)
type <- ifelse(.Platform$OS.type == 'windows', 'PSOCK', 'FORK')
cl <- makeCluster(4, type)
registerDoParallel(cl)
bt_tune <- tune_grid(
  object = bt_wf,
  resamples = bt_folds,
  grid = tree_grid[grid_row_start:(grid_row_start + 4), ],
  control = control_grid(
    save_pred = T,
    verbose = F,
    parallel_over = 'everything'
  )
)
stopCluster(cl)

result_file <- paste0(
  '../Results/model_fit_binary_outcome_version8_features_filtered/lightGBM/bt_tune_row_',
  grid_row_start,
  '-',
  (grid_row_start + 4),
  '(',
  total_size,
  ')_train_ratio_',
  train_ratio,
  '.rds'
)

saveRDS(bt_tune, file = result_file)



### Results integration-------------
library(tidymodels)
library(bonsai)

library(dplyr)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = T
)
index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')


bt_light <- boost_tree(
  trees = tune(),
  mtry = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_engine("lightgbm", objective = "multiclass", num_class = 2) %>%
  set_mode("classification")

bt_wf <- workflow() %>%
  add_formula(outcome ~ .) %>%
  add_model(bt_light)

measure_mat_ls <- list()
var_imp_ls <- list()
prob_mat_ls <- list()
for (train_ratio in c(0.6, 0.7, 0.8)) {
  train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
  test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index
  
  data_train <- data_df[train_index, ]
  data_test <- data_df[test_index, ]
  
  bt_tune <- NULL
  for (row_num in seq(1, 200, 5)) {
    result_file <- paste0(
      '../Results/model_fit_binary_outcome_version8_features_filtered/lightGBM/bt_tune_row_',
      row_num,
      '-',
      (row_num + 4),
      '(',
      200,
      ')_train_ratio_',
      train_ratio,
      '.rds'
    )
    bt_tune_tmp <- readRDS(result_file)
    bt_tune <- rbind(bt_tune, bt_tune_tmp)
  }
  
  
  bt_best <- select_best(x = bt_tune, metric = 'roc_auc')
  bt_fit <- bt_wf %>%
    finalize_workflow(bt_best) %>%
    fit(data_train)
  
  prob_train <- predict(bt_fit, new_data = data_train, type = 'prob')[, 2, drop =
                                                                        T]
  roc_train <- roc(data_train$outcome, prob_train)
  auc_train <- auc(roc_train)
  best_cutoff_train <- coords(roc_train, 'best')
  pred_class_train <- factor(
    ifelse(prob_train > as.numeric(best_cutoff_train[1]), 'birth', 'fail'),
    levels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_train <- caret::confusionMatrix(pred_class_train, data_train$outcome)
  measure_train <- c(auc_train,
                     conf_matrix_train$overall[c('Accuracy', 'Kappa')],
                     conf_matrix_train$byClass)
  names(measure_train)[1] <- 'Auc'
  
  measure_mat <- NULL
  measure_mat <- rbind(measure_mat, measure_train)
  prob_test <- predict(bt_fit, new_data  = data_test, type = "prob")[, 2, drop =
                                                                       T]
  
  # Evaluate the model
  library(pROC)
  roc_obj_test <- roc(data_test$outcome, prob_test)
  auc_test <- auc(roc_obj_test)
  best_cutoff_test <- coords(roc_obj_test, "best")
  pred_type_test <- factor(
    ifelse(prob_test > as.numeric(best_cutoff_test[1]), 'birth', 'fail'),
    levels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_test <- caret::confusionMatrix(pred_type_test, data_test$outcome)
  
  measure_test <- c(auc_test,
                    conf_matrix_test$overall[c('Accuracy', 'Kappa')],
                    conf_matrix_test$byClass)
  names(measure_test)[1] <- 'Auc'
  
  measure_mat <- rbind(measure_mat, measure_test)
  rownames(measure_mat) <- c('train', 'test')
  
  measure_mat <- cbind(measure_mat, rbind(as.matrix(bt_best[, 1:6]), as.matrix(bt_best[, 1:6])))
  measure_mat_ls[[toString(train_ratio)]] <- measure_mat
  prob_ls <- list(
    outcome_train = data_train$outcome,
    outcome_test = data_test$outcome,
    prob_train = prob_train,
    prob_test = prob_test
  )
  prob_mat_ls[[toString(train_ratio)]] <- prob_ls
}


measure_df <- do.call(rbind, measure_mat_ls) %>%
  as.data.frame() %>%
  mutate(train_raio = rep(c(0.6, 0.7, 0.8), each = 2),
         data_type = rep(c('train', 'test'), times = 3)) %>%
  select(train_raio, data_type, everything())

write.csv(measure_df, file = '../Results/model_performace_binary_outcome_version8_features_filtered_lightGBM.csv', row.names = F)

saveRDS(prob_mat_ls, file = '../Results/prob_pred_binary_version8_features_filtered/prob_lightGBM_ls.rds')



## GBM---------------
train_ratio <- commandArgs(trailingOnly = TRUE)

library(dplyr)
require(gbm)
library(caret)
library(doParallel)

data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')
loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1
data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = T
)

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')
train_index <- index_train_test_ls[[train_ratio]]$train_index
test_index <- index_train_test_ls[[train_ratio]]$test_index

data_train <- data_df[train_index, ]
data_test <- data_df[test_index, ]

gbmGrid <-  expand.grid(
  interaction.depth = c(7, 9, 11, 13),
  n.trees = (3:20) * 100,
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = 10
)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


type <- ifelse(.Platform$OS.type == 'windows', 'PSOCK', 'FORK')
cl <- makeCluster(5, type)
registerDoParallel(cl)

gbm_tuned <- train(
  outcome ~ .,
  data = data_train,
  method = "gbm",
  metric = 'ROC',
  tuneGrid = gbmGrid,
  trControl = ctrl,
  verbose = FALSE
)
stopCluster(cl)

result_file <- paste0(
  '../Results/model_fit_binary_outcome_version8_features_filtered/gbm_tuned_train_ratio_',
  train_ratio,
  '.rds'
)
saveRDS(gbm_tuned, file = result_file)




### Results integration-------------
library(dplyr)
library(pROC)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1
data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = T
)

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')
measure_mat_ls <- list()
confu_mat_ls <- list()
prob_mat_ls <- list()
for (train_ratio in c(0.6, 0.7, 0.8)) {
  result_file <- paste0(
    '../Results/model_fit_binary_outcome_version8_features_filtered/gbm_tuned_train_ratio_',
    train_ratio,
    '.rds'
  )
  gbm_tuned <- readRDS(result_file)
  
  train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
  test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index
  data_train <- data_df[train_index, ]
  data_test <- data_df[test_index, ]
  
  ## Train data results
  prob_train <- predict(gbm_tuned, type = 'prob')[, 2]
  roc_obj_train <- roc(data_train$outcome, prob_train)
  auc_train <- auc(roc_obj_train)
  
  best_cutoff_train <- coords(roc_obj_train, "best")
  
  pred_class_train <- factor(
    ifelse(prob_train > as.numeric(best_cutoff_train[1]), 1, 0),
    levels = c(0, 1),
    labels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_train <- confusionMatrix(pred_class_train, data_train$outcome)
  
  measure_train <- c(auc_train,
                     conf_matrix_train$overall[c('Accuracy', 'Kappa')],
                     conf_matrix_train$byClass)
  names(measure_train)[1] <- 'Auc'
  
  measure_mat <- NULL
  measure_mat <- rbind(measure_mat, measure_train)
  
  ## Test data results
  prob_test <- predict(gbm_tuned, newdata = data_test, type = "prob")[, 2]
  
  roc_obj_test <- roc(data_test$outcome, prob_test)
  auc_test <- auc(roc_obj_test)
  
  best_cutoff_test <- coords(roc_obj_test, "best")
  
  pred_class_test <- factor(
    ifelse(prob_test > as.numeric(best_cutoff_test[1]), 1, 0),
    levels = c(0, 1),
    labels = c('fail', 'birth'),
    ordered = T
  )
  conf_matrix_test <- confusionMatrix(pred_class_test, data_test$outcome)
  measure_test <- c(auc_test,
                    conf_matrix_test$overall[c('Accuracy', 'Kappa')],
                    conf_matrix_test$byClass)
  names(measure_test)[1] <- 'Auc'
  
  measure_mat <- rbind(measure_mat, measure_test)
  rownames(measure_mat) <- c('train', 'test')
  
  measure_mat_ls[[toString(train_ratio)]] <- measure_mat
  
  confu_mat_ls[[toString(train_ratio)]] <- list(train = conf_matrix_train$table, test = conf_matrix_test$table)
  
  prob_ls <- list(
    outcome_train = data_train$outcome,
    outcome_test = data_test$outcome,
    prob_train = prob_train,
    prob_test = prob_test
  )
  
  prob_mat_ls[[toString(train_ratio)]] <- prob_ls
  
}
measure_mat_ls
confu_mat_ls

measure_df <- do.call(rbind, measure_mat_ls) %>%
  as.data.frame() %>%
  mutate(train_raio = rep(c(0.6, 0.7, 0.8), each = 2),
         data_type = rep(c('train', 'test'), times = 3)) %>%
  dplyr::select(train_raio, data_type, everything())

write.csv(measure_df, file = '../Results/model_performace_binary_outcome_version8_features_filtered_GBM.csv', row.names = F)
saveRDS(prob_mat_ls, file = '../Results/prob_pred_binary_version8_features_filtered/prob_GBM_ls.rds')



## AdaBoost.M1 -----------------------------
train_ratio <- commandArgs(trailingOnly = TRUE)

library(dplyr)
library(adabag)
library(plyr)
library(doParallel)

data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')
loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1
data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = F
)


index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')
train_index <- index_train_test_ls[[train_ratio]]$train_index
test_index <- index_train_test_ls[[train_ratio]]$test_index

data_train <- data_df[train_index, ]
data_test <- data_df[test_index, ]


tuning_grid <- expand.grid(
  mfinal = seq(5, 30) * 100,
  maxdepth = c(3, 4, 5, 6),
  coeflearn = c('Breiman')
)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
type <- ifelse(.Platform$OS.type == 'windows', 'PSOCK', 'FORK')
cl <- makeCluster(5, type)
registerDoParallel(cl)

adaboost_tuned <- train(
  outcome ~ .,
  data = data_train,
  method = 'AdaBoost.M1',
  metric = 'ROC',
  tuneGrid = tuning_grid,
  trControl = ctrl,
  verbose = FALSE
)

stopCluster(cl)

result_file <- paste0(
  '../Results/model_fit_binary_outcome_version8_features_filtered/adaboost.M1_tuned_train_ratio_',
  train_ratio,
  '.rds'
)
saveRDS(adaboost_tuned, file = result_file)


### Results integration-------------
library(dplyr)
library(pROC)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')
loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1
data_df$outcome <- factor(
  data_df$outcome,
  levels = c(0, 1),
  labels = c('fail', 'birth'),
  ordered = F
)

index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')
measure_mat_ls <- list()
confu_mat_ls <- list()
prob_mat_ls <- list()
for (train_ratio in c(0.6, 0.7, 0.8)) {
  result_file <- paste0(
    '../Results/model_fit_binary_outcome_version8_features_filtered/adaboost.M1_tuned_train_ratio_',
    train_ratio,
    '.rds'
  )
  adaboost_tuned <- readRDS(result_file)
  train_index <- index_train_test_ls[[toString(train_ratio)]]$train_index
  test_index <- index_train_test_ls[[toString(train_ratio)]]$test_index
  data_train <- data_df[train_index, ]
  data_test <- data_df[test_index, ]
  
  ## Train data results
  prob_train <- predict(adaboost_tuned, newdata = data_train, type = 'prob')[, 2]
  roc_obj_train <- roc(data_train$outcome, prob_train)
  auc_train <- auc(roc_obj_train)
  
  best_cutoff_train <- coords(roc_obj_train, "best")
  pred_class_train <- factor(
    ifelse(prob_train > as.numeric(best_cutoff_train[1]), 1, 0),
    levels = c(0, 1),
    labels = c('fail', 'birth'),
    ordered = F
  )
  conf_matrix_train <- caret::confusionMatrix(pred_class_train, data_train$outcome)
  measure_train <- c(auc_train,
                     conf_matrix_train$overall[c('Accuracy', 'Kappa')],
                     conf_matrix_train$byClass)
  names(measure_train)[1] <- 'Auc'
  
  measure_mat <- NULL
  measure_mat <- rbind(measure_mat, measure_train)
  
  ## Test data results
  prob_test <- predict(adaboost_tuned, newdata = data_test, type = "prob")[, 2]
  roc_obj_test <- roc(data_test$outcome, prob_test)
  auc_test <- auc(roc_obj_test)
  best_cutoff_test <- coords(roc_obj_test, "best")
  pred_class_test <- factor(
    ifelse(prob_test > as.numeric(best_cutoff_test[1]), 1, 0),
    levels = c(0, 1),
    labels = c('fail', 'birth'),
    ordered = F
  )
  conf_matrix_test <- caret::confusionMatrix(pred_class_test, data_test$outcome)
  measure_test <- c(auc_test,
                    conf_matrix_test$overall[c('Accuracy', 'Kappa')],
                    conf_matrix_test$byClass)
  names(measure_test)[1] <- 'Auc'
  
  
  measure_mat <- rbind(measure_mat, measure_test)
  rownames(measure_mat) <- c('train', 'test')
  
  measure_mat_ls[[toString(train_ratio)]] <- measure_mat
  confu_mat_ls[[toString(train_ratio)]] <- list(train = conf_matrix_train$table, test = conf_matrix_test$table)
  
  prob_ls <- list(
    outcome_train = data_train$outcome,
    outcome_test = data_test$outcome,
    prob_train = prob_train,
    prob_test = prob_test
  )
  
  
  prob_mat_ls[[toString(train_ratio)]] <- prob_ls
}


measure_df <- do.call(rbind, measure_mat_ls) %>%
  as.data.frame() %>%
  mutate(train_raio = rep(c(0.6, 0.7, 0.8), each = 2),
         data_type = rep(c('train', 'test'), times = 3)) %>%
  dplyr::select(train_raio, data_type, everything())

write.csv(measure_df, file = '../Results/model_performace_binary_outcome_version8_features_filtered_AdaBoostM1.csv', row.names = F)

saveRDS(prob_mat_ls, file = '../Results/prob_pred_binary_version8_features_filtered/prob_AdaBoostM1_ls.rds')

