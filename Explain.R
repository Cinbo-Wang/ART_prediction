# Model Explaination -------------------------------

## Feature importance of random forest: 1-AUC----------------

library(dplyr)
library(pROC)
library(caret)
library(randomForest)
library(DALEXtra)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(data_df$outcome,
                          levels = c(0, 1),
                          labels = c('fail', 'birth'))
index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

measure_mat_ls <- list()
var_imp_ls <- list()
prob_mat_ls <- list()

train_ratio <- 0.8
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

## Result of Refit using the opitmal parameters
set.seed(200)
fit.rf <- randomForest(
  outcome ~ .,
  data = data_train,
  mtry = as.numeric(best_mtry),
  ntree = as.numeric(best_ntree)
)

## Model explain--
data_explain <- data_test
rf_exp <- explain_tidymodels(
  model = fit.rf,
  data = data_explain[, -which(names(data_explain) ==
                                 "outcome")],
  y = ifelse(data_explain$outcome == 'fail', 0, 1),
  label = 'Random Forest'
)
set.seed(1980)
fit.imp <- ingredients::feature_importance(
  x = rf_exp,
  type = 'raw',
  loss_function = DALEX::loss_one_minus_auc,
  B = 50
)

saveRDS(fit.imp, file = '../Results/model_fit_binary_outcome_version8_features_filtered/model_explain_Randomforest/fit.feature_imp_One_minus_AUC.rds')


### Plot the importance figure
require(ggplot2)
require(dplyr)
fit.imp <- readRDS(
  '../Results/model_fit_binary_outcome_version8_features_filtered/model_explain_Randomforest/fit.feature_imp_One_minus_AUC.rds'
)

plot(fit.imp, max_vars = 20, show_boxplots = T) +
  ggtitle("Variable importance")

fit.imp2 <- fit.imp
feature_name_map <- as.data.frame(read.csv('../Data/变量名修正v2.csv'))
old2new <- feature_name_map$New_name
names(old2new) <- feature_name_map$Old_name

fit.imp2$variable_new <- old2new[fit.imp2$variable]
fit.imp2$variable_new[is.na(fit.imp2$variable_new)] <- fit.imp2$variable[is.na(fit.imp2$variable_new)]
fit.imp2$variable <- fit.imp2$variable_new
fit.imp2$variable_new <- NULL

plot(
  fit.imp2,
  max_vars = 20,
  show_boxplots = T,
  bar_width = 8
) +
  ggtitle("Variable importance")


## Dataset Level -------------------
### CP, LP, AP: Female_age, P_basal, type_embryos_trans_--------------------
library(dplyr)
library(pROC)
library(caret)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(data_df$outcome,
                          levels = c(0, 1),
                          labels = c('fail', 'birth'))
index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

measure_mat_ls <- list()
var_imp_ls <- list()
prob_mat_ls <- list()


train_ratio <- 0.8
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


## Result of Refit using the optimal parameters
library(randomForest)
set.seed(200)
fit.rf <- randomForest(
  outcome ~ .,
  data = data_train,
  mtry = as.numeric(best_mtry),
  ntree = as.numeric(best_ntree)
)

## Model explanation --
data_explain <- data_test
library(DALEXtra)
rf_exp <- explain_tidymodels(
  model = fit.rf,
  data = data_explain[, -which(names(data_explain) ==
                                 "outcome")],
  y = ifelse(data_explain$outcome == 'fail', 0, 1),
  label = 'Randomforest'
)

var_imp_df <- read.csv(
  '../Results/Var_importance_One_minuse_AUC_binary_outcome_version8_features_filtered_randomforest.csv'
)
feature_name_map <- as.data.frame(read.csv('../Data/变量名修正.csv'))
new2old <- feature_name_map$Old_name
names(new2old) <- feature_name_map$New
var_imp_df$variable_old <- new2old[var_imp_df$variable]

old2new <- feature_name_map$New_name
names(old2new) <- feature_name_map$Old_name

features_imp <- c("age_female", 'P_basal')
data_model_explain_ls <- list()
for (old_name in features_imp) {
  print(paste0(
    old_name,
    "(location:",
    which(features_imp %in% old_name),
    ')'
  ))
  library(DALEX)
  library(DALEXtra)
  new_name <- old2new[old_name]
  num_sample <- NULL
  pdp_rf <- model_profile(
    rf_exp,
    variables = old_name,
    N = num_sample,
    type = 'partial',
    variable_type = 'numerical'
  )
  
  ldp_rf <- model_profile(
    rf_exp,
    variables = old_name,
    N = num_sample,
    type = 'conditional',
    variable_type = 'numerical'
  )
  alp_rf <- model_profile(
    rf_exp,
    variables = old_name,
    N = num_sample,
    type = 'accumulated',
    variable_type = 'numerical'
  )
  
  pdp_rf$agr_profiles$`_label_` = "Partial dependence"
  ldp_rf$agr_profiles$`_label_` = "Local dependence"
  alp_rf$agr_profiles$`_label_` = "Accumulated local"
  
  data_model_explain <- data.frame(
    x = c(
      pdp_rf$agr_profiles$`_x_`,
      ldp_rf$agr_profiles$`_x_`,
      alp_rf$agr_profiles$`_x_`
    ),
    y = c(
      pdp_rf$agr_profiles$`_yhat_`,
      ldp_rf$agr_profiles$`_yhat_`,
      alp_rf$agr_profiles$`_yhat_`
    ),
    line_group = c(
      pdp_rf$agr_profiles$`_label_`,
      ldp_rf$agr_profiles$`_label_`,
      alp_rf$agr_profiles$`_label_`
    )
  )
  data_model_explain_ls[[old_name]] <- data_model_explain
  
}



fig_ls <- list()
for (old_name in features_imp) {
  print(paste0(
    old_name,
    "(location:",
    which(features_imp %in% old_name),
    ')'
  ))
  library(DALEX)
  library(DALEXtra)
  new_name <- old2new[old_name]
  
  data_model_explain <- data_model_explain_ls[[old_name]]
  breaks <- seq(min(data_explain[, old_name]), max(data_explain[, old_name]), length.out = 101)
  height_hist <- max(table(cut(data_explain[, old_name], breaks = breaks)))
  
  scaling_factor <- ceiling(height_hist / max(data_model_explain$y))
  fig_dependence_single <-
    ggplot() +
    geom_line(
      data = data_model_explain,
      aes(
        x = x,
        y = y,
        group = line_group,
        color = line_group,
        linetype = line_group
      ),
      linewidth = 1
    ) +
    geom_histogram(
      data = data.frame(x = data_explain[, old_name]),
      aes(x = x, y = after_stat(count) / scaling_factor),
      bins = 100,
      alpha = 0.5,
      fill = 'gray'
    ) +
    scale_y_continuous(name = 'Probability',
                       sec.axis = sec_axis( ~ . * scaling_factor, name = "Frequency")) +
    scale_x_continuous(breaks = round(seq(
      min(data_model_explain$x),
      max(data_model_explain$x),
      length.out = 5
    ), digits = 2)) +
    labs(title = "", x = new_name, y = 'Probability') +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 17),
      panel.background = element_rect(fill = "white"),
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_line(color = "gray90"),
      axis.text.x = element_text(size = 15),
      axis.text.y = element_text(size = 15),
      axis.title.x = element_text(size = 20),
      axis.title.y = element_text(size = 18)
    )
  fig_ls[[old_name]] <- fig_dependence_single
}

pdf(
  '../Results/model_fit_binary_outcome_version8_features_filtered/model_explain_Randomforest/PD_LD_AL_profile_age_female_Pbasal.pdf',
  width = 9,
  height = 8
)
for (i in 1:length(fig_ls)) {
  print(fig_ls[[i]])
}
dev.off()



## Discrete feature
features_imp <- c('type_embryos_trans_')
var_imp_df <- read.csv(
  '../Results/Var_importance_One_minuse_AUC_binary_outcome_version8_features_filtered_randomforest.csv'
)
feature_name_map <- as.data.frame(read.csv('../Data/变量名修正.csv'))
old2new <- feature_name_map$New_name
names(old2new) <- feature_name_map$Old_name

for (old_name in features_imp) {
  print(old_name)
  library(DALEX)
  library(DALEXtra)
  new_name <- old_name
  num_sample <- NULL
  pdp_rf <- model_profile(
    rf_exp,
    variables = old_name,
    N = num_sample,
    type = 'partial',
    variable_type = 'categorical'
  )
  
  ldp_rf <- model_profile(
    rf_exp,
    variables = old_name,
    N = num_sample,
    type = 'conditional',
    variable_type = 'categorical'
  )
  alp_rf <- model_profile(
    rf_exp,
    variables = old_name,
    N = num_sample,
    type = 'accumulated',
    variable_type = 'categorical'
  )
  
  pdp_rf$agr_profiles$`_label_` = "Partial dependence"
  ldp_rf$agr_profiles$`_label_` = "Local dependence"
  alp_rf$agr_profiles$`_label_` = "Accumulated local"
  
  fig_dependence_single <- plot(pdp_rf, ldp_rf, alp_rf)
  
  data_fig <- data.frame(
    type = c(
      pdp_rf$agr_profiles$`_x_`,
      ldp_rf$agr_profiles$`_x_`,
      alp_rf$agr_profiles$`_x_`
    ),
    y = c(
      pdp_rf$agr_profiles$`_yhat_`,
      ldp_rf$agr_profiles$`_yhat_`,
      alp_rf$agr_profiles$`_yhat_`
    ),
    method = c(
      pdp_rf$agr_profiles$`_label_`,
      ldp_rf$agr_profiles$`_label_`,
      alp_rf$agr_profiles$`_label_`
    )
  )
  
  data_fig$type <- factor(
    data_fig$type,
    levels = c(
      'inferior',
      'inferior_inferior',
      'good',
      'good_inferior',
      'good_good'
    ),
    labels = c(
      '1 Inferior Embryo',
      '2 Inferior Embryos',
      '1 Good Embryo',
      '1 Good and 1 Inferior Embryo',
      '2 Good Embryos'
    )
  )
  
}


pdf(
  '../Results/model_fit_binary_outcome_version8_features_filtered/model_explain_Randomforest/PD_LD_AL_profile_type_embryos_trans.pdf',
  width = 8,
  height = 8
)
color_group <- c(
  `Accumulated local` = "#F8766D",
  `Local dependence` = "#00BA38",
  `Partial dependence` = "#619CFF"
)
ggplot(data_fig, aes(x = type, y = y, fill = method)) +
  geom_bar(
    stat = "identity",
    position = position_dodge(),
    alpha = 0.9,
    width = 0.5
  ) +
  labs(title = "", x = old2new[old_name], y = "Average prediction") +
  theme_minimal() +
  theme(
    legend.title = element_blank(),
    legend.position = 'bottom',
    legend.text = element_text(size = 13),
    legend.key.size = unit(2, "lines"),
    axis.text.x = element_text(angle = 10, size = 10),
    plot.title = element_text(hjust = 0.5),
    axis.text.y = element_text(size = 15),
    axis.title.x = element_text(size = 20),
    axis.title.y = element_text(size = 18)
  )
dev.off()



### Two-dim PD -----------------------
library(randomForest)
library(pdp)
library(dplyr)
library(ggplot2)
library(pROC)
library(caret)

data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')
loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(data_df$outcome,
                          levels = c(0, 1),
                          labels = c('fail', 'birth'))
index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

measure_mat_ls <- list()
var_imp_ls <- list()
prob_mat_ls <- list()


train_ratio <- 0.8
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


## Result of Refit using the opitmal parameters
set.seed(200)
fit.rf <- randomForest(
  outcome ~ .,
  data = data_train,
  mtry = as.numeric(best_mtry),
  ntree = as.numeric(best_ntree)
)

feature_name_map <- as.data.frame(read.csv('../Data/变量名修正.csv'))
old2new <- feature_name_map$New_name
names(old2new) <- feature_name_map$Old_name

features_ls <- list(c("thickness_intima", "E2_on_HCG_day"),
                    c("AMH", "E2_on_HCG_day"))
fig_result_ls <- list()
for (j in 1:length(features_ls)) {
  features <- features_ls[[j]]
  x <- data_train[, features[1]]
  y <- data_train[, features[2]]
  N_point <- 20
  if (length(unique(x)) > N_point) {
    x_disc <- seq(quantile(x, 0.05), quantile(x, 0.95), length.out = N_point)
  } else{
    x_disc <- sort(unique(x))
  }
  if (length(unique(y)) > N_point) {
    y_disc <- seq(quantile(y, 0.05), quantile(y, 0.95), length.out = N_point)
  } else{
    y_disc <- sort(unique(y))
  }
  
  pred.grid <- NULL
  for (x_tmp in x_disc) {
    for (y_tmp in y_disc) {
      pred.grid <- rbind(pred.grid, c(x_tmp, y_tmp))
    }
  }
  pred.grid <- as.data.frame(pred.grid)
  colnames(pred.grid) <- features
  pdp_result <- partial(fit.rf, pred.var = features, pred.grid = pred.grid)
  
  fig_result_ls[[j]] <- pdp_result
  
  
}


pdf(
  '../Results/model_fit_binary_outcome_version8_features_filtered/model_explain_Randomforest/2D_partial_dependence_to_paper.pdf',
  width = 8,
  height = 8
)
for (iter_num in positions) {
  pdp_result <- fig_result_ls[[iter_num]]
  features <- colnames(pdp_result)[1:2]
  fig <- autoplot(pdp_result, contour = TRUE) +
    labs(title = "", x = old2new[features[1]], y = old2new[features[2]]) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(size = 13),
      legend.text = element_text(size = 10),
      legend.key.size = unit(2, "lines"),
      panel.background = element_rect(fill = "white"),
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_line(color = "gray90"),
      axis.text.x = element_text(size = 15),
      axis.text.y = element_text(size = 15),
      axis.title.x = element_text(size = 20),
      axis.title.y = element_text(size = 18)
    )
  print(fig)
}
dev.off()




## Instance Level: Break-down-----------------
library(dplyr)
library(pROC)
library(caret)
data_df <- readRDS('../Data/data_clean_version8_features_filtered.rda')

loc_fail <- which(data_df$outcome <= 1)
data_df$outcome[loc_fail] <- 0
data_df$outcome[-loc_fail] <- 1

data_df$outcome <- factor(data_df$outcome,
                          levels = c(0, 1),
                          labels = c('fail', 'birth'))
index_train_test_ls <- readRDS('../Data/train_test_index_ls_type2_data_clean_version8.rds')

measure_mat_ls <- list()
var_imp_ls <- list()
prob_mat_ls <- list()


train_ratio <- 0.8
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


## Result of Refit using the optimal parameters
library(randomForest)
set.seed(200)
fit.rf <- randomForest(
  outcome ~ .,
  data = data_train,
  mtry = as.numeric(best_mtry),
  ntree = as.numeric(best_ntree)
)

## Model explanation --
data_explain <- data_test

library(DALEXtra)
rf_exp <- explain_tidymodels(
  model = fit.rf,
  data = data_test[, -which(names(data_test) ==
                              "outcome")],
  y = ifelse(data_test$outcome == 'fail', 0, 1),
  label = 'Randomforest'
)

feature_name_map <- as.data.frame(read.csv('../Data/变量名修正v2.csv'))
rownames(feature_name_map) <- feature_name_map$Old_name
value_type_embryos_trans <- c('single inferior',
                              'both inferior',
                              'single good',
                              '1 good and 1 inferior',
                              'both good')
names(value_type_embryos_trans) <- c('inferior',
                                     'inferior_inferior',
                                     'good',
                                     'good_inferior',
                                     'good_good')

# Fail
data_test[loc_explain, ]
rf_bd <- predict_parts(rf_exp, new_observation = data_test[loc_explain, -which(names(data_test) ==
                                                                                 "outcome")], type = "break_down")

rf_bd_new <- rf_bd
rownames(rf_bd_new) <- rf_bd_new$variable_name

rf_bd_new[rf_bd_new$variable_name, 'variable_name_new'] <- feature_name_map[rf_bd_new$variable_name, 'New_name']
rf_bd_new$variable_name_new[nrow(rf_bd)] <- ""
rf_bd_new$variable_name_new[1] <- "intercept"
rf_bd_new$variable_value[which(rf_bd_new$variable_name == 'type_embryos_trans_')] <-
  value_type_embryos_trans[rf_bd_new$variable_value[which(rf_bd_new$variable_name ==
                                                            'type_embryos_trans_')]]

rf_bd_new$variable_new <- paste0(rf_bd_new$variable_name_new, ' = ', rf_bd_new$variable_value)
rf_bd_new$variable_new[nrow(rf_bd)] <- 'prediction'
rf_bd_new$variable_new[1] <- "intercept"
rf_bd_new$variable <- rf_bd_new$variable_new
rf_bd_new$variable_name <- rf_bd_new$variable_name_new
rf_bd_new$variable_new <- NULL
rf_bd_new$variable_name_new <- NULL

# Adjust the order and display the results based on the absolute value of the contribution.
rf_bd_new2 <- rf_bd_new %>%
  arrange(desc(abs(contribution)))
rf_bd_new2 <- rbind(rf_bd_new2[rf_bd_new2$variable == 'intercept', ], rf_bd_new2[!(rf_bd_new2$variable %in% c('intercept', 'prediction')), ], rf_bd_new2[rf_bd_new2$variable ==
                                                                                                                                                           'prediction', ])
for (i in 2:(nrow(rf_bd_new2) - 1)) {
  rf_bd_new2$cumulative[i] <- rf_bd_new2$cumulative[i - 1] + rf_bd_new2$contribution[i]
  rf_bd_new2$position[i] <- rf_bd_new2$position[i - 1] - 1
}

fig_bd_fail <- plot(
  rf_bd_new2,
  max_features = 20,
  title = paste0("Break Down profile"),
  subtitle = paste0('True Outcome: ', data_test[loc_explain, 'outcome'])
)

rf_bd_fail <- rf_bd_new2

# Birth

rf_bd_birth <- predict_parts(rf_exp, new_observation = data_test[loc_explain, -which(names(data_test) ==
                                                                                       "outcome")], type = "break_down")
rf_bd_new <- rf_bd_birth
rownames(rf_bd_new) <- rf_bd_new$variable_name

rf_bd_new[rf_bd_new$variable_name, 'variable_name_new'] <- feature_name_map[rf_bd_new$variable_name, 'New_name']
rf_bd_new$variable_name_new[nrow(rf_bd_birth)] <- ""
rf_bd_new$variable_name_new[1] <- "intercept"
rf_bd_new$variable_value[which(rf_bd_new$variable_name == 'type_embryos_trans_')] <-
  value_type_embryos_trans[rf_bd_new$variable_value[which(rf_bd_new$variable_name ==
                                                            'type_embryos_trans_')]]
rf_bd_new$variable_new <- paste0(rf_bd_new$variable_name_new, ' = ', rf_bd_new$variable_value)
rf_bd_new$variable_new[nrow(rf_bd_birth)] <- 'prediction'
rf_bd_new$variable_new[1] <- "intercept"
rf_bd_new$variable <- rf_bd_new$variable_new
rf_bd_new$variable_name <- rf_bd_new$variable_name_new
rf_bd_new$variable_new <- NULL
rf_bd_new$variable_name_new <- NULL

# Adjust the order and display the results based on the absolute value of the contribution.
rf_bd_new2 <- rf_bd_new %>%
  arrange(desc(abs(contribution)))
rf_bd_new2 <- rbind(rf_bd_new2[rf_bd_new2$variable == 'intercept', ], rf_bd_new2[!(rf_bd_new2$variable %in% c('intercept', 'prediction')), ], rf_bd_new2[rf_bd_new2$variable ==
                                                                                                                                                           'prediction', ])
for (i in 2:(nrow(rf_bd_new2) - 1)) {
  rf_bd_new2$cumulative[i] <- rf_bd_new2$cumulative[i - 1] + rf_bd_new2$contribution[i]
  rf_bd_new2$position[i] <- rf_bd_new2$position[i - 1] - 1
}

fig_bd_birth <- plot(
  rf_bd_new2,
  max_features = 20,
  title = paste0("Break Down profile"),
  subtitle = paste0('True Outcome: ', data_test[loc_explain, 'outcome'])
)

rf_bd_birth <- rf_bd_new2


pdf(
  '../Results/model_fit_binary_outcome_version8_features_filtered/model_explain_Randomforest/Break_down_profile_fail_birth.pdf',
  width = 8,
  height = 8
)

print(fig_bd_fail)

print(fig_bd_birth)

dev.off()
