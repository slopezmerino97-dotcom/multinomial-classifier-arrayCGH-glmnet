getwd()
setwd("..\\CATS_project")
load("Data\\workspace_B4TM.RData")
ls()
library('caret')
library('glmnet')
# Hello
#This grid is for testing
#feature_sizes <- c(5, 10)
#alpha_grid <- c(0, 1)
#lambda_grid <- c(0.001, 0.01)
#This is the second grid, expanding the feature sizes,alpha and lambda values
#feature_sizes<-c(5,10,20,30)
#alpha_grid<-c(0,0.5,1)
#lambda_grid<-10^seq(-3,0,length.out=8)
#Let's try a 3rd grid to check consistency
feature_sizes<-c(15,20,25)
alpha_grid<-c(0,1)
lambda_grid<-c(0.001,0.002,0.005)
select_top_features <- function(df, n_features) {
  
  # keep the features without response variable
  x <- df[, colnames(df) != "Subgroup", drop = FALSE]
  y <- df$Subgroup
  
  # make sure that y is factor
  y <- as.factor(y)
  
  # make sure that all x's columns are numeric
  x[] <- lapply(x, function(col) as.numeric(as.character(col)))
  
  # compute variable importance
  imp <- filterVarImp(x = x, y = y)
  
  # if imp is empty then return character 0
  if (is.null(imp) || nrow(imp) == 0) {
    return(character(0))
  }
  
  # For multiclass classification compute mean imp feature
  mean_imp <- rowMeans(imp, na.rm = TRUE)
  
  # Sort features
  ranked_features <- names(sort(mean_imp, decreasing = TRUE))
  
  # return top n
  ranked_features[seq_len(min(n_features, length(ranked_features)))]
}
n_repeats<-100
outer_summary <- data.frame(
  repetition=integer(),
  outer_fold = integer(),
  best_n_features = numeric(),
  best_alpha = numeric(),
  best_lambda = numeric(),
  outer_accuracy = numeric(),
  stringsAsFactors = FALSE
)

all_outer_predictions <- data.frame(
  repetition=integer(),
  truth = character(),
  pred = character(),
  outer_fold = integer(),
  stringsAsFactors = FALSE
)
#saving the selected features (2185...)
selected_features_log <- data.frame(
  repetition = integer(),
  outer_fold = integer(),
  feature = character(),
  stringsAsFactors = FALSE
)

for (rep in 1:n_repeats){
  cat("\n============================\n")
  cat("REPETITION:", rep, "\n")
  cat("============================\n")
  
  set.seed(100+rep)
  outer_folds<-createFolds(data_model$Subgroup,k=3,returnTrain = TRUE)
  for (i in seq_along(outer_folds)) {
  
  train_outer <- data_model[outer_folds[[i]], , drop = FALSE]
  test_outer  <- data_model[-outer_folds[[i]], , drop = FALSE]
  
  inner_folds <- createFolds(train_outer$Subgroup, k = 5, returnTrain = TRUE)
  
  inner_results <- expand.grid(
    n_features = feature_sizes,
    alpha = alpha_grid,
    lambda = lambda_grid
  )
  
  inner_results$mean_accuracy <- NA_real_
  
  for (r in 1:nrow(inner_results)) {
    
    n_feat <- inner_results$n_features[r]
    alpha_val <- inner_results$alpha[r]
    lambda_val <- inner_results$lambda[r]
    
    inner_accuracies <- c()
    
    for (j in 1:length(inner_folds)) {
      
      inner_train <- train_outer[inner_folds[[j]], , drop = FALSE]
      inner_valid <- train_outer[-inner_folds[[j]], , drop = FALSE]
      
      top_features <- select_top_features(inner_train, n_feat)
      
      if (length(top_features) < 2) {
        next
      }
      
      x_inner_train <- as.matrix(inner_train[, top_features, drop = FALSE])
      y_inner_train <- inner_train$Subgroup
      
      x_inner_valid <- as.matrix(inner_valid[, top_features, drop = FALSE])
      y_inner_valid <- inner_valid$Subgroup
      
      if (ncol(x_inner_train) < 2 || ncol(x_inner_valid) < 2) {
        next
      }
      
      fit_inner <- glmnet(
        x = x_inner_train,
        y = y_inner_train,
        family = "multinomial",
        alpha = alpha_val,
        lambda = lambda_val,
        standardize = TRUE
      )
      
      pred_inner <- predict(fit_inner, newx = x_inner_valid, type = "class")
      pred_inner <- as.vector(pred_inner)
      
      acc_inner <- mean(pred_inner == y_inner_valid)
      
      inner_accuracies <- c(inner_accuracies, acc_inner)
    }
    
    if (length(inner_accuracies) == 0) {
      inner_results$mean_accuracy[r] <- NA_real_
    } else {
      inner_results$mean_accuracy[r] <- mean(inner_accuracies, na.rm = TRUE)
    }
  }
  
  best_n_row <- inner_results[
    which.max(replace(inner_results$mean_accuracy, is.na(inner_results$mean_accuracy), -Inf)),
  ]
  
  best_n_features <- best_n_row$n_features
  best_alpha <- best_n_row$alpha
  best_lambda <- best_n_row$lambda
  
  cat("REPETITION:", rep, "- OUTER FOLD:", i, "\n")
  cat("best_n_features:", best_n_features, "\n")
  cat("best_alpha:", best_alpha, "\n")
  cat("best_lambda:", best_lambda, "\n")
  
  final_features <- select_top_features(train_outer, best_n_features)
  cat("length(final_features):", length(final_features), "\n")
  cat("first final_features:", paste(head(final_features), collapse = ", "), "\n")
  
  if (length(final_features) < 2) {
    cat("Skipped: fewer than 2 final features\n\n")
    next
  }
  selected_features_log <- rbind(
    selected_features_log,
    data.frame(
      repetition = rep,
      outer_fold = i,
      feature = final_features,
      stringsAsFactors = FALSE
    )
  )
  
  x_train_final <- as.matrix(train_outer[, final_features, drop = FALSE])
  y_train_final <- train_outer$Subgroup
  
  x_test_final <- as.matrix(test_outer[, final_features, drop = FALSE])
  y_test_final <- test_outer$Subgroup
  
  cat("ncol(x_train_final):", ncol(x_train_final), "\n")
  cat("ncol(x_test_final):", ncol(x_test_final), "\n")
  
  if (ncol(x_train_final) < 2 || ncol(x_test_final) < 2) {
    cat("Skipped: fewer than 2 columns in final matrices\n\n")
    next
  }
  
  final_model <- glmnet(
    x = x_train_final,
    y = y_train_final,
    family = "multinomial",
    alpha = best_alpha,
    lambda = best_lambda,
    standardize = TRUE
  )
  
  pred_outer <- predict(final_model, newx = x_test_final, type = "class")
  pred_outer <- as.vector(pred_outer)
  
  outer_acc <- mean(pred_outer == y_test_final)
  
  cat("outer_accuracy:", outer_acc, "\n\n")
  
  outer_summary <- rbind(
    outer_summary,
    data.frame(
      repetition=rep,
      outer_fold = i,
      best_n_features = best_n_features,
      best_alpha = best_alpha,
      best_lambda = best_lambda,
      outer_accuracy = outer_acc
    )
  )
  
  all_outer_predictions <- rbind(
    all_outer_predictions,
    data.frame(
      repetition=rep,
      truth = as.character(y_test_final),
      pred = as.character(pred_outer),
      outer_fold = i,
      stringsAsFactors = FALSE
    )
  )
  }
}
mean(outer_summary$outer_accuracy)
sd(outer_summary$outer_accuracy)

rep_means<-aggregate(outer_accuracy~repetition,data=outer_summary,mean)
rep_means
mean(rep_means$outer_accuracy)
sd(rep_means$outer_accuracy)
save(outer_summary,rep_means,feature_sizes,alpha_grid,lambda_grid,file = "nestedCV_100reps_results_newgrid.RData")
write.csv2(outer_summary, "outer_summary_newgrid.csv", row.names = FALSE)
write.csv2(rep_means, "rep_means_newgrid.csv", row.names = FALSE)
summary_table <- data.frame(
  mean_accuracy = mean(rep_means$outer_accuracy),
  sd_accuracy = sd(rep_means$outer_accuracy),
  min_accuracy = min(rep_means$outer_accuracy),
  max_accuracy = max(rep_means$outer_accuracy)
)

summary_table
write.csv2(summary_table, "performance_summary_newgrid.csv", row.names = FALSE)

write.csv2(as.data.frame(table(outer_summary$best_n_features)),
           "feature_size_frequency_newgrid.csv",
           row.names = FALSE)

write.csv2(as.data.frame(table(outer_summary$best_alpha)),
           "alpha_frequency_newgrid.csv",
           row.names = FALSE)

write.csv2(as.data.frame(table(outer_summary$best_lambda)),
           "lambda_frequency_newgrid.csv",
           row.names = FALSE)
feature_selection_frequency <- as.data.frame(table(selected_features_log$feature))
colnames(feature_selection_frequency) <- c("feature", "Freq")

feature_selection_frequency <- feature_selection_frequency[
  order(feature_selection_frequency$Freq, decreasing = TRUE),
]

write.csv2(
  selected_features_log,
  "selected_features_log_newgrid.csv",
  row.names = FALSE
)

write.csv2(
  feature_selection_frequency,
  "feature_selection_frequency_newgrid.csv",
  row.names = FALSE
)
save.image("workspace_nestedCV_newgrid.RData")
